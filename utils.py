import tensorflow as tf
import numpy as np
import json, re

FLAGS = tf.flags.FLAGS

def load_data(filename, map_relations):
    print "Loading data:", filename
    f_in = open(filename)
    inp = f_in.readline()
    data = json.loads(inp)
    num_sent = 0
    cnt_multi_parents = 0
    for dialog in data:
        last_speaker = None
        turn = 0
        for edu in dialog["edus"]:
            edu["text_raw"] = edu["text"] + " "
            text = edu["text"]
            
            while text.find("http") >= 0:
                i = text.find("http")
                j = i
                while (j < len(text) and text[j] != ' '): j += 1
                text = text[:i] + " [url] " + text[j + 1:]
            
            invalid_chars = ["/", "\*", "^", ">", "<", "\$", "\|", "=", "@"]
            for ch in invalid_chars:
                text = re.sub(ch, "", text)
            tokens = []
            cur = 0
            for i in range(len(text)):
                if text[i] in "',?.!()\": ":
                    if (cur < i):
                        tokens.append(text[cur:i])
                    if text[i] != " ":
                        if len(tokens) == 0 or tokens[-1] != text[i]:
                            tokens.append(text[i])
                    cur = i + 1
            if cur < len(text):
                tokens.append(text[cur:])
            tokens = [token.lower() for token in tokens]
            for i, token in enumerate(tokens):
                if re.match("\d+", token): 
                    tokens[i] = "[num]"
            edu["tokens"] = tokens
            
            if edu["speaker"] != last_speaker:
                last_speaker = edu["speaker"]
                turn += 1
            edu["turn"] = turn
        have_relation = {}
        relations = []
        for relation in dialog["relations"]:
            if (relation["x"], relation["y"]) in have_relation: 
                continue
            relations.append(relation)
            have_relation[(relation["x"], relation["y"])] = True
        dialog["relations"] = relations
        for relation in dialog["relations"]:
            if not relation["type"] in map_relations:
                map_relations[relation["type"]] = len(map_relations)
            relation["type"] = map_relations[relation["type"]]
        def cmp_relation(a, b):
            if a["x"] == b["x"] and a["y"] == b["y"]: return 0
            if a["y"] < b["y"] or (a["y"] == b["y"] and a["x"] < b["x"]): return -1
            return 1
        dialog["relations"] = sorted(dialog["relations"], cmp=lambda a,b:cmp_relation(a, b))
        cnt = [0] * len(dialog["edus"])
        for r in dialog["relations"]:
            cnt[r["y"]] += 1
        for i in range(len(dialog["edus"])):
            if cnt[i] > 1:
                cnt_multi_parents += 1        
    f_in.close()
    cnt_edus, cnt_relations, cnt_relations_backward = 0, 0, 0
    for dialog in data:
        cnt_edus += len(dialog["edus"])
        cnt_relations += len(dialog["relations"])
        for r in dialog["relations"]:
            if r["x"] > r["y"]:
                cnt_relations_backward += 1
    print "%d dialogs, %d edus, %d relations, %d backward relations" % \
        (len(data), cnt_edus, cnt_relations, cnt_relations_backward)
    print "%d edus have multiple parents" % cnt_multi_parents    
        
    return data

def build_vocab(data):
    print "Building vocabulary..."
    vocab = {}
    for dialog in data:
        for edu in dialog["edus"]:
            sentences = [edu["tokens"]]
            for sentence in sentences:
                for token in sentence:
                    if token in vocab:
                        vocab[token] += 1
                    else:
                        vocab[token] = 1
    vocab_list = ["UNK", "PAD", "EOS"] + sorted(vocab, key=vocab.get, reverse=True)

    print("Loading word vectors...")
    vectors = {}
    f_in = open(FLAGS.word_vector)
    for line in f_in:
        line = line.split()
        vectors[line[0]] = map(float, line[1:])
    f_in.close()
    embed = []
    cnt_pretrained = 0
    vocab_list_major = []
    for i, word in enumerate(vocab_list):
        if i > FLAGS.vocab_size and (not word in vectors):
            continue
        vocab_list_major.append(word)
        if word in vectors:
            embed.append(vectors[word])
            cnt_pretrained += 1
        else:
            embed.append(np.zeros(FLAGS.dim_embed_word, dtype=np.float32))
            
    embed = np.array(embed, dtype=np.float32)
    print "Pre-trained vectors: %d/%d" % (cnt_pretrained, len(embed))
    return vocab_list_major, embed    

def preview_data(data, pred, map_relations, vocab, idx):
    map_relations_inv = {}
    for item in map_relations:
        map_relations_inv[map_relations[item]] = item
    for i, dialog in enumerate(data):
        print idx
        idx += 1
        for j, edu in enumerate(dialog["edus"]):
            print j, edu["speaker"], ":", 
            for token in edu["tokens"]:
                if not token in vocab:
                    print "UNK(%s)" % token, 
                else:
                    print token,
            print
        print "ground truth:"
        for relation in dialog["relations"]:
            print relation["x"], relation["y"], map_relations_inv[relation["type"]]
        print "predicted:"
        for relation in pred[i]:
            print relation[0], relation[1], map_relations_inv[relation[2]]
        
        std = np.zeros((len(dialog["edus"]), len(dialog["edus"])))
        for relation in dialog["relations"]:
            std[relation["x"]][relation["y"]] = relation["type"] + 1
        cnt_cor = 0
        for relation in pred[i]:
            if std[relation[0]][relation[1]] == relation[2] + 1:
                cnt_cor += 1
        if len(pred[i]) > 0 and len(dialog["relations"]) > 0:
            prec = cnt_cor * 1. / len(pred[i])
            recall = cnt_cor * 1. / len(dialog["relations"])
            if prec == 0 and recall == 0:
                f1 = 0
            else:
                f1 = 2 * prec * recall / (prec + recall)
            print "f1:", f1
        print
    return idx
        
def update_buffer(output_feed, input_feed, feed):
    return (output_feed + feed[0], dict(input_feed.items() + feed[1].items()))    

def init_grad(params):
    return [
        np.zeros(shape=param.shape)
        for param in params
    ]

def get_batches(data, batch_size, sort=True):
    if sort:
        data = sorted(data, key=lambda dialog: len(dialog['edus']))
    while (len(data[0]['edus']) == 0): data = data[1:]
    batches = []
    for i in range(len(data) / batch_size + bool(len(data) % batch_size)):
        batches.append(data[i * batch_size : (i + 1) * batch_size])
    return batches