"""
Baseline

Bi-GRU encoder -> MLP -> MST/ILP
"""
import tensorflow as tf
import numpy as np
import os, json, random, time, re, math
from Sentence_Encoder import Sentence_Encoder
from utils import load_data, build_vocab
from os import path as fp
from ilp import load_scip_output, mk_zimpl_input, dump_scores_to_dat_files

if not os.environ.has_key("CUDA_VISIBLE_DEVICES"): 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_boolean("train", False, "is training")
tf.flags.DEFINE_string("scip_path", "~/SCIPOptSuite-6.0.0-Linux/bin/scip", "path to SCIP")
tf.flags.DEFINE_boolean("load_checkpoint", True, "load checkpoint")
tf.flags.DEFINE_string("word_vector", "../glove/glove.6B.100d.txt", "word vector")
tf.flags.DEFINE_string("model_dir", "model", "model directory")
tf.flags.DEFINE_string("log_dir", "log", "log directory")
tf.flags.DEFINE_string("method", "mst", "method mst/ilp/greedy")
tf.flags.DEFINE_integer("max_edu_dist", 20, "maximum distance between two related edus")
tf.flags.DEFINE_integer("dim_embed_word", 100, "dimension of word embedding")
tf.flags.DEFINE_integer("dim_traditional", 3, "dimension of traditional features")
tf.flags.DEFINE_integer('num_epochs', 50, 'number of epochs')
tf.flags.DEFINE_integer("num_units", 256, "number of units in RNN encoder")
tf.flags.DEFINE_integer("num_layers", 1, "number of RNN layers")
tf.flags.DEFINE_integer("num_relations", 16, "number of relations")
tf.flags.DEFINE_integer("batch_size", 4, "batch size")
tf.flags.DEFINE_integer("vocab_size", 1000, "vocabulary size")
tf.flags.DEFINE_float("positive_lb", 0.5, "lower bound of positive samples")
tf.flags.DEFINE_float("keep_prob", 0.5, "probability to keep units in dropout")
tf.flags.DEFINE_float("learning_rate", 0.1, "learning rate")
tf.flags.DEFINE_float("learning_rate_decay", 0.98, "learning rate decay factor")

def padding(sent, l):
    return sent + ["EOS"] + ["PAD"] * (l - len(sent) - 1)

def get_batched_data_test(dialog, edges_candidate):
    data = get_batches([dialog], 1, is_test=True)[0]
    
    x, y, traditional_features = [], [], []
    for sample in edges_candidate:
        x.append(sample["i"])
        y.append(sample["j"])
        traditional_features.append(sample["traditional_features"])
    
    return {
        "text_string": data["text_string"],
        "len_word": data["len_word"],
        "len_doc": data["len_doc"],
        "len_sent": data["len_sent"],
        "x_bi": np.array(x),
        "y_bi": np.array(y),
        "traditional_features_bi": np.array(traditional_features),
        "x_multi": np.array(x),
        "y_multi": np.array(y),
        "traditional_features_multi": np.array(traditional_features),
    }  
    
def get_batches(data, batch_size, positive_lb=0, is_test=False):
    batches = []
    
    random.shuffle(data)
    
    text_string, len_word, len_sent, len_doc = [], [], [], []
    x_bi, y_bi, relation_bi, traditional_features_bi = [], [], [], []
    x_multi, y_multi, relation_multi, traditional_features_multi, idx = [], [], [], [], []
    
    for i in range(len(data) / batch_size + bool(len(data) % batch_size)):
        max_num_sent, max_num_word = 0, 0
        for dialog in data[i * batch_size : (i + 1) * batch_size]:
            max_num_sent = max(max_num_sent, len(dialog["edus"]))
            for edu in dialog["edus"]:
                max_num_word = max(max_num_word, len(edu["tokens"]))
        max_num_word += 1
                
        sample_pos, sample_neg = [], []
        for dialog in data[i * batch_size : (i + 1) * batch_size]:
            text_string.append([])
            len_word.append([])
            for edu in dialog["edus"]:
                text_string[-1].append(padding(edu["tokens"], max_num_word))
                len_word[-1].append(len(edu["tokens"]) + 1)
            length = len(dialog["edus"])                  
            for i in range(max_num_sent - length):
                text_string[-1].append(["POS"] * max_num_word)
                len_word[-1].append(0)                          
            len_doc.append(length)
            
            std = np.zeros((length, length), dtype=np.int32)
            for relation in dialog["relations"]:
                std[relation["x"]][relation["y"]] = relation["type"] + 1
            if FLAGS.method == "ilp":
                for i in range(length):
                    for j in range(length):
                        if j - i <= FLAGS.max_edu_dist:
                            sample = {
                                "x": (len(text_string) - 1) * max_num_sent + i,
                                "y": (len(text_string) - 1) * max_num_sent + j,
                                "relation": std[i][j],
                                "traditional_features": [
                                    bool(dialog["edus"][i]["speaker"] == dialog["edus"][j]["speaker"]),
                                    bool(dialog["edus"][i]["turn"] == dialog["edus"][j]["turn"]),
                                    j - i
                                ]
                            }
                            if std[i][j] > 0:
                                sample_pos.append(sample)
                            else:
                                sample_neg.append(sample)                
            else:
                for j in range(length):
                    for i in range(j):
                        if j - i <= FLAGS.max_edu_dist:
                            sample = {
                                "x": (len(text_string) - 1) * max_num_sent + i,
                                "y": (len(text_string) - 1) * max_num_sent + j,
                                "relation": std[i][j],
                                "traditional_features": [
                                    bool(dialog["edus"][i]["speaker"] == dialog["edus"][j]["speaker"]),
                                    bool(dialog["edus"][i]["turn"] == dialog["edus"][j]["turn"]),
                                    j - i
                                ]
                            }
                            if std[i][j] > 0:
                                sample_pos.append(sample)
                            else:
                                sample_neg.append(sample)                
                                
        if positive_lb == 0:
            max_negative = len(sample_neg)
        else:
            max_negative = int(len(sample_pos) / positive_lb - len(sample_pos))

        random.shuffle(sample_neg)
        sample_neg = sample_neg[:max_negative]

        samples = sample_pos + sample_neg
        random.shuffle(samples)
        
        if not is_test:
            for sample in samples:
                x_bi.append(sample["x"])
                y_bi.append(sample["y"])
                relation_bi.append(bool(sample["relation"] > 0))
                traditional_features_bi.append(sample["traditional_features"])
                if sample["relation"] > 0:
                    x_multi.append(sample["x"])
                    y_multi.append(sample["y"])
                    relation_multi.append(sample["relation"] - 1)
                    traditional_features_multi.append(sample["traditional_features"])
                    idx.append(len(x_bi) - 1)
            if len(x_bi) == 0 or len(x_multi) == 0: 
                text_string, len_word, len_sent, len_doc = [], [], [], []
                x_bi, y_bi, relation_bi, traditional_features_bi = [], [], [], []
                x_multi, y_multi, relation_multi, traditional_features_multi, idx = [], [], [], [], []  
                continue
            
        batches.append({
            "text_string": np.array(text_string),
            "len_word": np.array(len_word),
            "len_sent": np.array(len_sent),
            "len_doc": np.array(len_doc),
            "x_bi": np.array(x_bi),
            "y_bi": np.array(y_bi),
            "relation_bi": np.array(relation_bi),
            "traditional_features_bi": np.array(traditional_features_bi),
            "x_multi": np.array(x_multi),
            "y_multi": np.array(y_multi),
            "relation_multi": np.array(relation_multi),
            "traditional_features_multi": np.array(traditional_features_multi),
            "idx": np.array(idx),
        })
        text_string, len_word, len_sent, len_doc = [], [], [], []
        x_bi, y_bi, relation_bi, traditional_features_bi = [], [], [], []
        x_multi, y_multi, relation_multi, traditional_features_multi, idx = [], [], [], [], []            

    return batches
    
def test_mst():
    cnt_golden, cnt_pred, cnt_cor_bi, cnt_cor_multi = 0, 0, 0, 0
    
    _cnt_golden = np.zeros(200)
    _cnt_pred = np.zeros(200)
    _cnt_cor_bi = np.zeros(200)
    _cnt_cor_multi = np.zeros(200)
    f1_bi = np.zeros(200)
    f1_multi = np.zeros(200)    
    
    for i, dialog in enumerate(data_test):
        if len(dialog["edus"]) == 1: continue
        edges_candidate = []
        for j in range(len(dialog["edus"])):
            if FLAGS.method == "greedy" or j == 0 or j > 0 and dialog["edus"][j]["turn"] != dialog["edus"][j - 1]["turn"]: # inter-turn
                for i in range(len(dialog["edus"])):
                    if i < j and j - i <= FLAGS.max_edu_dist:
                        edges_candidate.append({
                            "traditional_features": [
                                bool(dialog["edus"][i]["speaker"] == dialog["edus"][j]["speaker"]),
                                bool(dialog["edus"][i]["turn"] == dialog["edus"][j]["turn"]),
                                j - i
                            ],
                            "i": i,
                            "j": j
                        })                    
            else: # intra-turn
                i = j - 1
                edges_candidate.append({
                    "traditional_features": [
                        bool(dialog["edus"][i]["speaker"] == dialog["edus"][j]["speaker"]),
                        bool(dialog["edus"][i]["turn"] == dialog["edus"][j]["turn"]),
                        1
                    ],
                    "i": i,
                    "j": j
                })

        batch = get_batched_data_test(dialog, edges_candidate)
        r, weight = sentence_encoder.infer(batch)
        best_weight = [-1] * len(dialog["edus"])
        edge = [None] * len(dialog["edus"])
        for i, e in enumerate(edges_candidate):
            e["type"] = r[i]
            e["weight"] = weight[i]
            if e["weight"] > best_weight[e["j"]]:
                best_weight[e["j"]] = e["weight"]
                edge[e["j"]] = i
                
        pred = []
        cnt_in = [0] * len(dialog["edus"])
        for i in range(len(dialog["edus"])):
            if edge[i] is not None:
                idx = edge[i]
                cnt_in[edges_candidate[idx]["j"]] += 1
                pred.append((edges_candidate[idx]["i"], edges_candidate[idx]["j"], edges_candidate[idx]["type"]))
        root_pred = 0
        while (cnt_in[root_pred] > 0): root_pred += 1
        cnt_in = [0] * len(dialog["edus"])
        for relation in dialog["relations"]:
            cnt_in[relation["y"]] += 1
        
        cnt_pred += 1
        for i in range(len(dialog["edus"])):
            if cnt_in[i] == 0:
                cnt_golden += 1
                _cnt_golden[i + 1] += 1
        if cnt_in[root_pred] == 0:
            cnt_cor_bi += 1
            cnt_cor_multi += 1
            
        relation_types = np.zeros((len(dialog["edus"]), len(dialog["edus"])), dtype=np.int32)
        for relation in dialog["relations"]:
            relation_types[relation["x"]][relation["y"]] = relation["type"] + 1
            if relation["y"] > relation["x"]:
                _cnt_golden[relation["y"] - relation["x"]] += 1            

            last = -1
            for rr in dialog["relations"]:
                if rr["y"] == relation["x"] and rr["x"] < relation["x"] and rr["x"] > last:
                    last = rr["x"]
            if last == -1:
                last = relation["x"]
                
            cnt_golden += 1
            
        _cnt_pred[1] += 1
        if cnt_in[0] == 0:
            _cnt_cor_bi[1] += 1
            _cnt_cor_multi[1] += 1            
            
        for r in pred:
            cnt_pred += 1
            if relation_types[r[0]][r[1]] > 0:
                cnt_cor_bi += 1
                if r[2] == relation_types[r[0]][r[1]] - 1:
                    cnt_cor_multi += 1
                    
            last = -1
            for rr in dialog["relations"]:
                if rr["y"] == r[0] and rr["x"] < r[0] and rr["x"] > last:
                    last = rr["x"]
            if last == -1:
                last = r[0]
                    
            _cnt_pred[r[1] - r[0]] += 1
            if relation_types[r[0]][r[1]] > 0:
                _cnt_cor_bi[r[1] - r[0]] += 1
                if relation_types[r[0]][r[1]] == r[2] + 1:
                    _cnt_cor_multi[r[1] - r[0]] += 1 
    
    for i in range(20):
        prec = _cnt_cor_bi[i] * 1. / _cnt_pred[i]
        recall = _cnt_cor_bi[i] * 1. / _cnt_golden[i]
        f1_bi[i] = 2. * prec * recall / (prec + recall)
        prec = _cnt_cor_multi[i] * 1. / _cnt_pred[i]
        recall = _cnt_cor_multi[i] * 1. / _cnt_golden[i]
        f1_multi[i] = 2. * prec * recall / (prec + recall)    
             
    prec_bi = cnt_cor_bi * 1. / cnt_pred
    recall_bi = cnt_cor_bi * 1. / cnt_golden
    f1_bi = 2 * prec_bi * recall_bi / (prec_bi + recall_bi)
    prec_multi = cnt_cor_multi * 1. / cnt_pred
    recall_multi = cnt_cor_multi * 1. / cnt_golden
    f1_multi = 2 * prec_multi * recall_multi / (prec_multi + recall_multi)
    return [
        prec_bi,
        recall_bi,
        f1_bi,
        prec_multi,
        recall_multi,
        f1_multi
    ]
    
def test_ilp():
    cnt_golden, cnt_pred, cnt_cor_bi, cnt_cor_multi = 0, 0, 0, 0
    
    for i, dialog in enumerate(data_test):
        if len(dialog["edus"]) == 1: continue
        edges_candidate = []
        
        n = len(dialog["edus"])
        att_mat = np.zeros((n, n))
        lab_tsr = np.zeros((n, n, FLAGS.num_relations))
        
        for i in range(len(dialog["edus"])):
            for j in range(len(dialog["edus"])):
                edges_candidate.append({
                    "traditional_features": [
                        bool(dialog["edus"][i]["speaker"] == dialog["edus"][j]["speaker"]),
                        bool(dialog["edus"][i]["turn"] == dialog["edus"][j]["turn"]),
                        j - i
                    ],
                    "i": i,
                    "j": j
                })         
                
        batch = get_batched_data_test(dialog, edges_candidate)
        
        r, weight_bi, weight_multi = sentence_encoder.infer(batch)
        
        cur = 0
        for i in range(len(dialog["edus"])):
            for j in range(len(dialog["edus"])):
                att_mat[i][j] = weight_bi[cur]
                for k in range(FLAGS.num_relations):
                    lab_tsr[i][j][k] = weight_multi[cur][k]
                cur += 1
                   
        # Prepare ZIMPL template and data
        dump_scores_to_dat_files(dialog, att_mat, lab_tsr, "raw")
        input_path = mk_zimpl_input(dialog)
        
        # Run SCIP
        param_path = fp.join('scip.parameters')
        output_path = fp.join("./tmp", 'output.scip')
        
        os.system(FLAGS.scip_path + " " \
            + " -f " + input_path \
            + " -s " + param_path \
            + " > " + output_path)
        
        relation_types = np.zeros((len(dialog["edus"]), len(dialog["edus"])), dtype=np.int32)
        for relation in dialog["relations"]:
            relation_types[relation["x"]][relation["y"]] = relation["type"] + 1
            cnt_golden += 1
            
        pred= load_scip_output(dialog, output_path)        
        for r in pred:
            cnt_pred += 1
            if relation_types[r[0]][r[1]] > 0:
                cnt_cor_bi += 1
                if r[2] == relation_types[r[0]][r[1]] - 1:
                    cnt_cor_multi += 1
        
    prec_bi = cnt_cor_bi * 1. / cnt_pred
    recall_bi = cnt_cor_bi * 1. / cnt_golden
    f1_bi = 2 * prec_bi * recall_bi / (prec_bi + recall_bi)
    prec_multi = cnt_cor_multi * 1. / cnt_pred
    recall_multi = cnt_cor_multi * 1. / cnt_golden
    f1_multi = 2 * prec_multi * recall_multi / (prec_multi + recall_multi) 
    return [
        prec_bi,
        recall_bi,
        f1_bi,
        prec_multi,
        recall_multi,
        f1_multi
    ]

map_relations = {}
data_train = load_data('../data/STAC/train.json', map_relations)
data_test = load_data('../data/STAC/test.json', map_relations)
vocab, embed = build_vocab(data_train)
print "Dataset sizes: %d/%d" % (len(data_train), len(data_test))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
with sess.as_default():
    sentence_encoder = Sentence_Encoder(sess, FLAGS, embed)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    global_step_inc_op = global_step.assign(global_step + 1)   
    epoch = tf.Variable(0, name='epoch', trainable=False)
    epoch_inc_op = epoch.assign(epoch + 1)     

    saver = tf.train.Saver(
        write_version=tf.train.SaverDef.V2,
        max_to_keep=None, 
        pad_step_number=True, 
        keep_checkpoint_every_n_hours=1.0
    )        
    
    summary_list = [
        "loss", "loss_bi", "loss_multi",
        "prec_bi", "recall_bi", "f1_bi",
        "prec_multi", "recall_multi", "f1_multi",
    ]
    summary_num = len(summary_list)
    len_output_feed = 7

    if FLAGS.train:
        if FLAGS.load_checkpoint and tf.train.get_checkpoint_state(FLAGS.model_dir):
            print "Reading model parameters from %s" % FLAGS.model_dir
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir))
        else:
            print "Created model with fresh parameters"
            sentence_encoder.initialize(vocab=vocab)
            sess.run(tf.global_variables_initializer())            
        print "Trainable variables:"
        for var in tf.trainable_variables():
            print var

        train_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, "train"))
        test_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, "test"))
        summary_placeholders = [tf.placeholder(tf.float32) for i in range(summary_num)]
        summary_op = [tf.summary.scalar(summary_list[i], summary_placeholders[i]) for i in range(summary_num)]

        test_batches = get_batches(data_test, FLAGS.batch_size)
        
        best_test = [0, 0]
        while epoch.eval() < FLAGS.num_epochs:
            epoch_inc_op.eval()
            summary_steps = 0
            
            train_batches = get_batches(data_train, FLAGS.batch_size, FLAGS.positive_lb)

            start_time = time.time()
            s = np.zeros(len_output_feed)
            for batch in train_batches:
                ops = sentence_encoder.step(batch, train=True)
                
                for i in range(len_output_feed):
                    s[i] += ops[i]
                        
                summary_steps += 1
                global_step_inc_op.eval()
                global_step_val = global_step.eval()                         
                
            summary_sum = [
                s[0] / summary_steps,
                s[1] / summary_steps,
                s[2] / summary_steps,
                s[5] * 1. / s[3], # prec_bi
                s[5] * 1. / s[4], # recall_bi
                2. * s[5] * 1. / s[3] * s[5] * 1. / s[4] / (s[5] * 1. / s[3] + s[5] * 1. / s[4]), # f1_bi
                s[6] * 1. / s[3], # prec_multi
                s[6] * 1. / s[4], # recall_multi
                2. * s[6] * 1. / s[3] * s[6] * 1. / s[4] / (s[6] * 1. / s[3] + s[6] * 1. / s[4]), # f1_multi,
            ]
            
            print "Epoch %s" % epoch.eval()
            for k in range(3, summary_num):
                print "  Train %s: %.5lf" % (
                    summary_list[k], 
                    summary_sum[k]
                )
            summaries = sess.run(summary_op, feed_dict=dict(zip(summary_placeholders, summary_sum)))
            for s in summaries:
                train_writer.add_summary(summary=s, global_step=global_step_val)                        
            s = np.zeros(len_output_feed)
            summary_steps = 0
            start_time = time.time() 
                    
            saver.save(sess, "%s/checkpoint" % FLAGS.model_dir, global_step=global_step_val)                          

            if FLAGS.method == "ilp":
                res_test = test_ilp()
            else:
                res_test = test_mst()
            summary_sum = [0] * 3 + res_test
            
            summaries = sess.run(summary_op, feed_dict=dict(zip(summary_placeholders, summary_sum)))
            for s in summaries:
                test_writer.add_summary(summary=s, global_step=global_step_val)
                          
            for k in range(3, 9):
                print "  Test %s: %.5lf" % (summary_list[k], summary_sum[k])

            if summary_sum[8] > best_test[1]:
                best_test[0] = summary_sum[5]
                best_test[1] = summary_sum[8]
            
            sentence_encoder.learning_rate_decay_op.eval()
            
            print "  Best test: %.3f %.3f" % (best_test[0], best_test[1])
    
    else:
        print "Reading model parameters from %s" % FLAGS.model_dir 
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir))
        if FLAGS.method == "ilp":
            res_test = test_ilp()
        else:
            res_test = test_mst()