import tensorflow as tf
import numpy as np
import os
from libs import DropoutWrapper

class NonStructured_Encoder():
    def __init__(self, sess, FLAGS, embed, num_units=None, scope="Sentence_Encoder"):
        self.sess = sess
        self.dim_embed_word = FLAGS.dim_embed_word
        self.num_units = num_units if (num_units is not None) else FLAGS.num_units
        self.num_layers = FLAGS.num_layers
        self.train_keep_prob = FLAGS.keep_prob
        
        self.fixed_noise = tf.placeholder(tf.int32)
        self.keep_prob = tf.placeholder_with_default(1.0, ())
        self.learning_rate = tf.placeholder(tf.float32)
        if FLAGS.use_adam:
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)      
        
        self.recurrent_noise_in, self.recurrent_noise_out, self.recurrent_noise = [], [], None
    
        with tf.variable_scope(scope):        
            self._build_embedding(embed)
            self._build_input()
            self._build_encoders()
        
        self.params = []
        for var in tf.trainable_variables():
            if var.name.find(os.path.join(tf.contrib.framework.get_name_scope(), scope)) == 0:
                self.params.append(var)
                
        self.grad_out = tf.gradients(
            tf.concat([self.enc_text, self.enc_text_cont], axis=-1), 
            self.params, 
            tf.concat([self.grad_enc_text, self.grad_enc_text_cont], axis=-1)
        )
        self.grad_out[0] = tf.convert_to_tensor(self.grad_out[0])
        self.grad_in = [
            tf.placeholder(tf.float32, param.shape)
            for param in self.params
        ]
        self.train_op = self.optimizer.apply_gradients(zip(self.grad_in, self.params))           
        
    def _build_embedding(self, embed):
        self.symbol2index = tf.contrib.lookup.MutableHashTable(
            key_dtype=tf.string,
            value_dtype=tf.int64,
            default_value=0,
            shared_name="in_table",
            name="in_table",
            checkpoint=True)
        self.index2symbol = tf.contrib.lookup.MutableHashTable(
            key_dtype=tf.int64,
            value_dtype=tf.string,
            default_value="UNK",
            shared_name="out_table",
            name="out_table",
            checkpoint=True)
        self.embed = tf.get_variable("word_embedding", dtype=tf.float32, initializer=embed)  
        
    def _build_input(self):
        with tf.variable_scope("input"):
            self.num_posts = tf.placeholder(tf.int32, (None,), "num_posts")
            self.text_string = tf.placeholder(tf.string, (None, None, None), "text_string")
            self.text = tf.nn.embedding_lookup(
                self.embed, self.symbol2index.lookup(self.text_string))
            self.text_len = tf.placeholder(tf.int32, (None, None,), "text_len")
            
    def _build_encoders(self):
        with tf.variable_scope("encoders"):
            self.enc_text = self._build_encoder(
                tf.reshape(
                    self.text, 
                    [tf.shape(self.text)[0] * tf.shape(self.text)[1], tf.shape(self.text)[2], self.dim_embed_word]
                ),
                tf.reshape(self.text_len, [-1]),
                self.dim_embed_word, 
                True,
                "enc_text"
            )
            self.enc_text_cont = tf.reshape(
                self._build_encoder(
                    tf.reshape(
                        self.enc_text,
                        [tf.shape(self.text)[0], tf.shape(self.text)[1], self.num_units]
                    ),
                    self.num_posts,
                    self.num_units,
                    False,
                    "enc_text_cont"
                ),
                [-1, self.num_units]
            )
            self.grad_enc_text = tf.placeholder(tf.float32, self.enc_text.shape)
            self.grad_enc_text_cont = tf.placeholder(tf.float32, self.enc_text_cont.shape)
            
    def _build_encoder(self, inputs, length, input_size, use_biencoder, scope, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            if use_biencoder:
                cell_fw, cell_bw = self._build_biencoder_cell(input_size)
                outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=inputs,
                    sequence_length=length,
                    dtype=tf.float32
                )
                enc_state = []            
                for i in range(self.num_layers):
                    enc_state.append(tf.concat([states[0][i],states[1][i]], axis=-1))                
                return enc_state[-1]
            else:
                cell = self._build_cell(self.num_units, input_size)
                outputs, states = tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=inputs,
                    sequence_length=length,
                    dtype=tf.float32
                )
                return outputs
    
    def _build_cell(self, num_units, input_size):
        cells = []
        for i in range(self.num_layers):
            self.recurrent_noise_in.append(tf.placeholder(tf.float32, (1, input_size)))
            dropout = DropoutWrapper(
                tf.contrib.rnn.GRUCell(num_units), self.keep_prob,
                input_size=input_size, dtype=tf.float32,
                noise_input=self.recurrent_noise_in[-1],
                fixed_noise=self.fixed_noise
            )
            self.recurrent_noise_out.append(dropout.recurrent_input_noise)
            cells.append(dropout)
        
        return tf.contrib.rnn.MultiRNNCell(cells)
    
    def _build_biencoder_cell(self, input_size):
        cell_fw = self._build_cell(self.num_units / 2, input_size)
        cell_bw = self._build_cell(self.num_units / 2, input_size)
        return cell_fw, cell_bw        
        
    def initialize(self, vocab):
        op_in = self.symbol2index.insert(
            tf.constant(vocab), tf.constant(range(len(vocab)), dtype=tf.int64))
        op_out = self.index2symbol.insert(
            tf.constant(range(len(vocab)), dtype=tf.int64), tf.constant(vocab))
        self.sess.run([op_in, op_out])        

    def format_data(self, data):
        def padding(sent, l):
            return sent + ["EOS"] + ["PAD"] * (l - len(sent) - 1)

        length = 0
        for dialog in data:
            for text in dialog:
                length = max(length, len(text))
        length += 1
        
        text_string, text_len = [], []
        for dialog in data:
            text_string.append([])
            text_len.append([])
            for text in dialog:
                text_string[-1].append(padding(text, length))
                text_len[-1].append(len(text) + 1)
    
        return {
            "text_string": np.array(text_string),
            "text_len": np.array(text_len)
        }
        
    def get_gradients(self, data, num_posts, grad_enc_text, grad_enc_text_cont, buffered=False):
        data = self.format_data(data)
        input_feed = {
            self.text_string: data["text_string"],
            self.text_len: data["text_len"],
            self.num_posts: num_posts,                
            self.grad_enc_text: grad_enc_text,
            self.grad_enc_text_cont: grad_enc_text_cont,                
            self.keep_prob: self.train_keep_prob,
            self.fixed_noise: 1
        }
        for i in range(len(self.recurrent_noise)):
            input_feed[self.recurrent_noise_in[i]] = self.recurrent_noise[i]
        if buffered:
            return ([self.grad_out], input_feed)
        else:
            return self.sess.run(self.grad_out, input_feed)
        
    def train(self, grad, learning_rate, buffered=False):
        input_feed = {}
        for i in range(len(grad)):
            input_feed[self.grad_in[i]] = grad[i]
        input_feed[self.learning_rate] = learning_rate
        if buffered:
            return ([self.train_op], input_feed)
        else:
            self.sess.run(self.train_op, input_feed)

    def infer(self, data, num_posts, is_train, buffered=False):
        data = self.format_data(data)
        input_feed = {
            self.text_string: data["text_string"],
            self.text_len: data["text_len"],
            self.num_posts: num_posts, 
            self.fixed_noise: 0
        }
        for noise in self.recurrent_noise_in:
            input_feed[noise] = np.zeros(noise.shape)
        if is_train:
            input_feed[self.keep_prob] = self.train_keep_prob
        output_feed = [self.enc_text, self.enc_text_cont, self.recurrent_noise_out]
        if buffered:
            return (output_feed, input_feed)
        else:
            return self.sess.run(output_feed, input_feed)
