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
            
        # add a fake root
        if FLAGS.method == "ilp":
            for edu in dialog["edus"]:
                edu["turn"] += 1
            dialog["edus"] = [{
                "speaker": "fake_root",
                "turn": 0,
                "tokens": ["[root]"]
            }]  + dialog["edus"]
            cnt = [0] * len(dialog["edus"])
            root = 1    
            for r in dialog["relations"]:
                r["x"] += 1
                r["y"] += 1
                cnt[r["y"]] += 1
            for i in range(1, len(dialog["edus"])):
                if cnt[i] == 0:
                    dialog["relations"].append({
                        "x": 0,
                        "y": i,
                        "type": FLAGS.num_relations - 1
                    })     
            
        dialog["relations"] = sorted(dialog["relations"], cmp=lambda a,b:cmp_relation(a, b))
        cnt = [0] * len(dialog["edus"])
        for r in dialog["relations"]:
            cnt[r["y"]] += 1
        for i in range(len(dialog["edus"])):
            if cnt[i] > 1:
                cnt_multi_parents += 1
    f_in.close()
    cnt_edus, cnt_relations = 0, 0
    for dialog in data:
        cnt_edus += len(dialog["edus"])
        cnt_relations += len(dialog["relations"])
    print "%d dialogs, %d edus, %d relations" % (len(data), cnt_edus, cnt_relations)
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
