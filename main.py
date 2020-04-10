import tensorflow as tf
import numpy as np
import os, random, time
from Model import Model
from utils import load_data, build_vocab, preview_data

if not os.environ.has_key("CUDA_VISIBLE_DEVICES"): 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_boolean("is_train", False, "train model")

tf.flags.DEFINE_integer("display_interval", 5, "step interval to display information")
tf.flags.DEFINE_boolean("show_predictions", False, "show predictions in the test stage")
tf.flags.DEFINE_boolean("preview_data", False, "preview data")

tf.flags.DEFINE_string("data", "STAC", "data")
tf.flags.DEFINE_string("word_vector", "../glove/glove.6B.100d.txt", "word vector")
tf.flags.DEFINE_string("model_dir", "./model", "model directory")
tf.flags.DEFINE_string("log_dir", "./log", "log directory")
tf.flags.DEFINE_integer("valid_ratio", 0.1, "ratio of valid set")
tf.flags.DEFINE_integer("vocab_size", 1000, "vocabulary size")

tf.flags.DEFINE_integer("max_edu_dist", 20, "maximum distance between two related edus") 
tf.flags.DEFINE_integer("dim_embed_word", 100, "dimension of word embedding")
tf.flags.DEFINE_integer("dim_embed_relation", 100, "dimension of relation embedding")
tf.flags.DEFINE_integer("dim_feature_bi", 4, "dimension of binary features")

tf.flags.DEFINE_boolean("use_adam", False, "use adam optimizer")
tf.flags.DEFINE_boolean("use_structured", True, "use structured encoder")
tf.flags.DEFINE_boolean("use_speaker_attn", True, "use speaker highlighting mechanism")
tf.flags.DEFINE_boolean("use_shared_encoders", False, "use shared encoders")
tf.flags.DEFINE_boolean("use_random_structured", False, "use random structured repr.")
tf.flags.DEFINE_boolean("use_traditional", True, "use traditional features")

tf.flags.DEFINE_integer("num_units", 256, "number of hidden units")
tf.flags.DEFINE_integer("num_layers", 1, "number of RNN layers in encoders")
tf.flags.DEFINE_integer("num_relations", 16, "number of relation types")
tf.flags.DEFINE_integer("batch_size", 4, "batch size")
tf.flags.DEFINE_float("regularizer_scale", 1e-9, "regularizer scale")
tf.flags.DEFINE_float("keep_prob", 0.5, "probability to keep units in dropout")
tf.flags.DEFINE_float("learning_rate", 0.1, "learning rate")
tf.flags.DEFINE_float("learning_rate_decay", 0.98, "learning rate decay factor")

def get_batches(data, batch_size, sort=True):
    if sort:
        data = sorted(data, key=lambda dialog: len(dialog["edus"]))
    while (len(data[0]["edus"]) == 0): data = data[1:]
    batches = []
    for i in range(len(data) / batch_size + bool(len(data) % batch_size)):
        batches.append(data[i * batch_size : (i + 1) * batch_size])
    return batches
    
def get_summary_sum(s, length):
    loss_bi = s[0] / length
    loss_multi = s[1] / length
    prec_bi = s[4] * 1. / s[3]
    recall_bi = s[4] * 1. / s[2]
    f1_bi = 2 * prec_bi * recall_bi / (prec_bi + recall_bi)
    prec_multi = s[5] * 1. / s[3]
    recall_multi = s[5] * 1. / s[2]
    f1_multi = 2 * prec_multi * recall_multi / (prec_multi + recall_multi)
    return [ 
        loss_bi, loss_multi,
        prec_bi, recall_bi, f1_bi, 
        prec_multi, recall_multi, f1_multi
    ]    
    
map_relations = {}
data_train = load_data("../" + FLAGS.data + "/train.json", map_relations)
data_test = load_data("../" + FLAGS.data + "/test.json", map_relations)
valid_size = int(FLAGS.valid_ratio * len(data_train))
data_valid = data_train[-valid_size:]
data_train = data_train[:-valid_size]
vocab, embed = build_vocab(data_train)
print "Dataset sizes: %d/%d/%d" % (len(data_train), len(data_test), len(data_valid))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
with sess.as_default():
    model = Model(sess, FLAGS, embed, data_train)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    global_step_inc_op = global_step.assign(global_step + 1)    
    epoch = tf.Variable(0, name="epoch", trainable=False)
    epoch_inc_op = epoch.assign(epoch + 1)

    saver = tf.train.Saver(
        write_version=tf.train.SaverDef.V2,
        max_to_keep=None, 
        pad_step_number=True, 
        keep_checkpoint_every_n_hours=1.0
    )        
    
    summary_list = [
        "loss_bi", "loss_multi",
        "prec_bi", "recall_bi", "f1_bi",
        "prec_multi", "recall_multi", "f1_multi"
    ]
    summary_num = len(summary_list)
    len_output_feed = 6

    if FLAGS.is_train:
        if tf.train.get_checkpoint_state(FLAGS.model_dir):
            print "Reading model parameters from %s" % FLAGS.model_dir
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir))
        else:
            print "Created model with fresh parameters"
            sess.run(tf.global_variables_initializer())  
            model.initialize(vocab)          
            
        print "Trainable variables:"
        for var in tf.trainable_variables():
            print var

        train_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, "train"))
        valid_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, "valid"))
        test_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, "test"))
        summary_placeholders = [tf.placeholder(tf.float32) for i in range(summary_num)]
        summary_op = [tf.summary.scalar(summary_list[i], summary_placeholders[i]) for i in range(summary_num)]
        
        train_batches = get_batches(data_train, FLAGS.batch_size)
        valid_batches = get_batches(data_valid, FLAGS.batch_size)
        test_batches = get_batches(data_test, FLAGS.batch_size)
        
        best_test_f1 = [0] * 2
        while True:
            epoch_inc_op.eval()
            summary_steps = 0
            
            random.shuffle(train_batches)
            start_time = time.time()
            s = np.zeros(len_output_feed)
            
            for batch in train_batches:
                ops = model.step(batch, is_train=True)
                    
                for i in range(len_output_feed):
                    s[i] += ops[i]
                        
                summary_steps += 1
                global_step_inc_op.eval()
                global_step_val = global_step.eval()         
                if global_step_val % FLAGS.display_interval == 0:
                    print "epoch %d, global step %d (%.4fs/step):" % (
                        epoch.eval(), global_step_val, 
                        (time.time() - start_time) * 1. / summary_steps
                    )
                    summary_sum = get_summary_sum(s, summary_steps)
                    for k in range(summary_num):
                        print "  train %s: %.5lf" % (
                            summary_list[k], 
                            summary_sum[k]
                        )
                        
                    print "  best test f1:", best_test_f1[0], best_test_f1[1]
                        
            summary_sum = get_summary_sum(s, len(train_batches))            
            summaries = sess.run(summary_op, feed_dict=dict(zip(summary_placeholders, summary_sum)))
            for s in summaries:
                train_writer.add_summary(summary=s, global_step=epoch.eval())                        
            print "epoch %d (learning rate %.5lf)" % \
                (epoch.eval(), model.learning_rate.eval())
            for k in range(summary_num):
                print "  train %s: %.5lf" % (summary_list[k], summary_sum[k])                   
            
            s = np.zeros(len_output_feed)
            for batch in valid_batches:
                ops = model.step(batch)
                for i in range(len_output_feed):
                    s[i] += ops[i]
            summary_sum = get_summary_sum(s, len(valid_batches))
            summaries = sess.run(summary_op, feed_dict=dict(zip(summary_placeholders, summary_sum)))
            for s in summaries:
                valid_writer.add_summary(summary=s, global_step=epoch.eval())
            for k in range(summary_num):
                print "  valid %s: %.5lf" % (summary_list[k], summary_sum[k])                              
            
            s = np.zeros(len_output_feed)
            random.seed(0)
            for batch in test_batches:
                ops = model.step(batch)
                for i in range(len_output_feed):
                    s[i] += ops[i]
            summary_sum = get_summary_sum(s, len(test_batches))
            summaries = sess.run(summary_op, feed_dict=dict(zip(summary_placeholders, summary_sum)))
            for s in summaries:
                test_writer.add_summary(summary=s, global_step=epoch.eval())            
            for k in range(summary_num):
                print "  test %s: %.5lf" % (summary_list[k], summary_sum[k]) 
                
            if summary_sum[-1] > best_test_f1[1]:
                best_test_f1[0] = summary_sum[-4]
                best_test_f1[1] = summary_sum[-1]
            
            print "  best test f1:", best_test_f1[0], best_test_f1[1]
            
            model.learning_rate_decay_op.eval()             
            
            saver.save(sess, "%s/checkpoint" % FLAGS.model_dir, global_step=epoch.eval())                                            
    else:
        print "Reading model parameters from %s" % FLAGS.model_dir 
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir))
        
        test_batches = get_batches(data_test, 1, sort=False)
    
        s = np.zeros(len_output_feed)
        random.seed(0)
        idx = 0
        cnt_golden, cnt_pred, cnt_cor_bi, cnt_cor_multi = [], [], [], []
        for k, batch in enumerate(test_batches):
            if len(batch[0]["edus"]) == 1: 
                continue
            
            ops = model.step(batch)
            for i in range(len_output_feed):
                s[i] += ops[i]
                
            if FLAGS.show_predictions:
                idx = preview_data(batch, ops[-1], map_relations, vocab, idx)
            
            cnt_golden.append(ops[2])    
            cnt_pred.append(ops[3])    
            cnt_cor_bi.append(ops[4])    
            cnt_cor_multi.append(ops[5])    
                
        summary_sum = get_summary_sum(s, len(test_batches))
        
        print cnt_golden
        print cnt_pred
        print cnt_cor_bi
        print cnt_cor_multi
        print "Test:"
        for k in range(summary_num):
            print "  test %s: %.5lf" % (summary_list[k], summary_sum[k])   
