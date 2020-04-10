import tensorflow as tf
import numpy as np
import os
from libs import DropoutWrapper

class Structured_Encoder():
    def __init__(self, sess, FLAGS, scope="DAG_RNN"):
        self.sess = sess
        self.num_units = FLAGS.num_units
        self.num_layers = FLAGS.num_layers
        self.num_relations = FLAGS.num_relations
        self.dim_embed_relation = FLAGS.dim_embed_relation
        self.train_keep_prob = FLAGS.keep_prob
        
        self.fixed_noise = tf.placeholder(tf.int32)        
        self.keep_prob = tf.placeholder_with_default(1.0, ())
        self.learning_rate = tf.placeholder(tf.float32)
        if FLAGS.use_adam:
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)      
        
        with tf.variable_scope(scope):        
            self._build_embedding()
            self._build_input()
            self._build_encoder()
            
        self.params = []
        for var in tf.trainable_variables():
            if var.name.find(os.path.join(tf.contrib.framework.get_name_scope(), scope)) == 0:
                self.params.append(var)
                
        # out
        self.gradients = tf.gradients(self.result, self.params, self.grad_result)
        self.gradients[0] = tf.convert_to_tensor(self.gradients[0])
        self.grad_input = tf.gradients(self.result, [self.parent, self.current], self.grad_result)
        
        self.grad_in = [ tf.placeholder(tf.float32, param.shape) for param in self.params ]
        self.train_op = self.optimizer.apply_gradients(zip(self.grad_in, self.params))            
        
    def _build_embedding(self):
        self.embed = tf.get_variable(
            "relation_embedding", (self.num_relations + 1, self.dim_embed_relation), 
            dtype=tf.float32, initializer=tf.zeros_initializer
        )  
        
    def _build_input(self):
        with tf.variable_scope("input"):
            self.parent = tf.placeholder(tf.float32, (None, self.num_units), "parent")
            self.relation = tf.placeholder(tf.int32, (None,), "relation")
            self.relation_embed = tf.nn.embedding_lookup(self.embed, self.relation)
            self.current = tf.placeholder(tf.float32, (None, self.num_units), "current")
            
    def _build_encoder(self):
        with tf.variable_scope("encoder"):
            self.recurrent_noise_in, self.recurrent_noise_out, self.recurrent_noise = [], [], None
            self.recurrent_noise_in.append(tf.placeholder(tf.float32, (1, self.dim_embed_relation + self.num_units)))       
            dropout = DropoutWrapper(
                tf.contrib.rnn.GRUCell(self.num_units), self.keep_prob,
                input_size=self.dim_embed_relation+self.num_units, dtype=tf.float32,
                noise_input=self.recurrent_noise_in[-1],
                fixed_noise=self.fixed_noise
            )
            self.recurrent_noise_out.append(dropout.recurrent_input_noise)      
            self.cell = dropout
            
            self.result = self.cell.__call__(
                tf.concat([self.relation_embed, self.current], axis=-1),
                self.parent
            )[1]
            self.grad_result = tf.placeholder(tf.float32, self.result.shape)
            
    def get_gradients(self, grad_result, parent, current, relation, buffered=False):
        output_feed = [self.gradients, self.grad_input]
        input_feed = {
            self.grad_result: grad_result,
            self.parent: parent,
            self.current: current,
            self.relation: relation,
            self.keep_prob: self.train_keep_prob,
            self.fixed_noise: 1
        }
        for i in range(len(self.recurrent_noise)):
            input_feed[self.recurrent_noise_in[i]] = self.recurrent_noise[i]        
        if buffered:
            return (output_feed, input_feed)
        else:
            gradients = self.sess.run(output_feed, input_feed)
            return gradients[0], gradients[1][0], gradients[1][1]
        
    def train(self, grad, learning_rate, buffered=False):
        input_feed = {}
        for i in range(len(grad)):
            input_feed[self.grad_in[i]] = grad[i]
        input_feed[self.learning_rate] = learning_rate
        if buffered: 
            return ([self.train_op], input_feed)
        else:
            self.sess.run(self.train_op, input_feed)
   
    def infer(self, data, fixed_noise, is_train=False, buffered=False):
        input_feed = {
            self.parent: data["parent"],
            self.relation: data["relation"],
            self.current: data["current"],
            self.fixed_noise: fixed_noise
        }
        if fixed_noise and (self.recurrent_noise is not None):
            for i in range(len(self.recurrent_noise)):
                input_feed[self.recurrent_noise_in[i]] = self.recurrent_noise[i]            
        else:
            for noise in self.recurrent_noise_in:
                input_feed[noise] = np.zeros(noise.shape)        
        if is_train:
            input_feed[self.keep_prob] = self.train_keep_prob        
        output_feed = [self.result, self.recurrent_noise_out]
        if buffered:
            return (output_feed, input_feed)
        else:
            return self.sess.run(output_feed, input_feed)
