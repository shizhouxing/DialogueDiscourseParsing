import tensorflow as tf
import numpy as np
import random, os
from NonStructured_Encoder import NonStructured_Encoder
from Structured_Encoder import Structured_Encoder
from utils import update_buffer, init_grad
from libs import dropout

class Agent():
    def __init__(self, sess, FLAGS, embed, scope, is_multi, encoders=None):
        self.sess = sess
        self.is_multi = is_multi
        
        self.num_relations = FLAGS.num_relations
        self.num_units = FLAGS.num_units
        self.dim_feature_bi = FLAGS.dim_feature_bi
        self.use_structured = FLAGS.use_structured
        self.use_speaker_attn = FLAGS.use_speaker_attn     
        self.dim_state = 4 * self.num_units + (self.dim_feature_bi if FLAGS.use_traditional else 0)
        self.regularizer_scale = FLAGS.regularizer_scale
        self.train_keep_prob = FLAGS.keep_prob                       
        
        self.fixed_noise = tf.placeholder(tf.int32)                
        self.keep_prob = tf.placeholder_with_default(1.0, ())
        self.learning_rate = tf.placeholder(tf.float32)
        if FLAGS.use_adam:
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)        
        
        with tf.variable_scope(scope):
            self._build_input()
            self._build_policy_network()
            if encoders:
                self.ns_encoder = encoders[0]
                self.s_encoder_attn = encoders[1]
                self.s_encoder_general = encoders[2]
            else:
                self._build_encoders(FLAGS, embed)
        
        self.params_policy_network = []
        self.params = []
        for var in tf.trainable_variables():
            if var.name.find(os.path.join(tf.contrib.framework.get_name_scope(), os.path.join(scope, "policy_network"))) == 0:
                self.params_policy_network.append(var)
            if var.name.find(os.path.join(tf.contrib.framework.get_name_scope(), scope)) == 0:
                self.params.append(var)
        
        self._build_gradients()
        
        self.grad_policy_in = [
            tf.placeholder(tf.float32, param.shape)
            for param in self.params_policy_network
        ]
        self.train_op = self.optimizer.apply_gradients(zip(self.grad_policy_in, self.params_policy_network))
        
    def get_policy(self, state, mask=None):
        input_feed = { self.state: state }
        if not self.is_multi:
            input_feed[self.mask] = mask
        policy = self.sess.run(self.policy, input_feed)
        if not self.is_multi:
            policy = policy * mask
        return policy

    def get_gradients(self, state, golden, mask=None):
        input_feed = {
            self.state: state,
            self.golden: golden
        }
        output_feed = [self.loss, self.grad_policy_out, self.grad_state_out]
        if not self.is_multi:
            input_feed[self.mask] = mask
        return self.sess.run(output_feed, input_feed)
            
    def clear_gradients(self):
        self.grad_policy = init_grad(self.params_policy_network)
        if self.use_structured:
            self.grad_s_encoder_attn = init_grad(self.s_encoder_attn.params)
            self.grad_s_encoder_general = init_grad(self.s_encoder_general.params)
        self.grad_ns_encoder = init_grad(self.ns_encoder.params)
        
    def train(self, learning_rate, buffered=False):
        output_feed, input_feed = [], {}
        
        # update policy network
        input_feed = {}
        for i in range(len(self.grad_policy)):
            input_feed[self.grad_policy_in[i]] = self.grad_policy[i]
        input_feed[self.learning_rate] = learning_rate
        output_feed.append(self.train_op)
            
        # update structured encoder
        if self.use_structured:
            output_feed, input_feed = update_buffer(
                output_feed, input_feed, 
                self.s_encoder_general.train(self.grad_s_encoder_general, learning_rate, buffered=True)
            )
            if self.use_speaker_attn:
                output_feed, input_feed = update_buffer(
                    output_feed, input_feed,
                    self.s_encoder_attn.train(self.grad_s_encoder_attn, learning_rate, buffered=True) 
                )
                
        # update ns encoder
        output_feed, input_feed = update_buffer(
            output_feed, input_feed,
            self.ns_encoder.train(self.grad_ns_encoder, learning_rate, buffered=True)
        )
                
        if buffered:
            return (output_feed, input_feed)
        else:
            self.sess.run(output_feed, input_feed)

    def _build_gradients(self):
        if self.is_multi:
            self.loss = -tf.reduce_mean(tf.reduce_sum(
                tf.log(self.policy) * tf.one_hot(self.golden, self.num_relations), axis=-1))
        else:
            self.loss = -tf.reduce_mean(tf.reduce_sum(
                tf.log(self.policy) * tf.one_hot(self.golden, tf.shape(self.state)[1]) * self.mask, 
                axis=-1
            ))
        self.grad_policy_out = tf.gradients(self.loss, self.params_policy_network)
        self.grad_state_out = tf.gradients(self.loss, self.state)[0]      
        
    def _build_input(self):
        if self.is_multi:
            self.state = tf.placeholder(tf.float32, (None, self.dim_state), name="state")
        else:
            self.state = tf.placeholder(tf.float32, (None, None, self.dim_state), name="state")
            self.mask = tf.placeholder(tf.float32, (None, None), name="mask")
        self.golden = tf.placeholder(tf.int32, (None,), name="golden")  
        
    def _build_policy_network(self):
        with tf.variable_scope("policy_network"):
            h = tf.layers.dense(
                self.state, self.num_units * 2, activation=tf.tanh,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer_scale),
                bias_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer_scale)
            )
            if self.is_multi:
                self.policy = tf.nn.softmax(tf.layers.dense(h, self.num_relations))
            else:
                self.policy = self.softmax_with_mask(
                    tf.reduce_sum(tf.layers.dense(h, 1), axis=-1),
                    self.mask
                )        
        
    def _build_encoders(self, FLAGS, embed):
        num_units = self.num_units
                    
        self.ns_encoder = NonStructured_Encoder(self.sess, FLAGS, embed, num_units=num_units)
        
        if self.use_structured:
            self.s_encoder_general = Structured_Encoder(self.sess, FLAGS, scope="Structured_Encoder_general")
            if self.use_speaker_attn:
                self.s_encoder_attn = Structured_Encoder(self.sess, FLAGS, scope="Structured_Encoder_attn")
            else:
                self.s_encoder_attn = self.s_encoder_general
        else:
            self.s_encoder_general, self.s_encoder_attn = None, None
        
    def softmax_with_mask(self, h, mask):
        exp_with_mask = tf.exp(h * mask) * mask
        s = tf.reduce_sum(exp_with_mask, axis=-1)
        return tf.transpose(tf.transpose(exp_with_mask) / s) + (1 - mask)        
