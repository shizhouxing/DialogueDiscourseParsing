import tensorflow as tf

class Sentence_Encoder():
    def __init__(self, sess, FLAGS, embed):
        self.sess = sess
        self.num_units = FLAGS.num_units
        self.num_hidden_units = self.num_units * 2
        self.num_layers = FLAGS.num_layers
        self.num_relations = FLAGS.num_relations
        self.train_keep_prob = FLAGS.keep_prob
        self.dim_embed_word = FLAGS.dim_embed_word
        self.dim_traditional = FLAGS.dim_traditional
        self.max_gradient_norm = 5.0
        self.method = FLAGS.method
        
        self.keep_prob = tf.placeholder_with_default(1.0, ())
        self.learning_rate = tf.Variable(
            float(FLAGS.learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_reset_op = self.learning_rate.assign(
            tf.constant(FLAGS.learning_rate))                                             
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * FLAGS.learning_rate_decay)
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        
        with tf.variable_scope("Sentence_Encoder"):        
            self._build_embedding(embed)
            self._build_input()
            self._build_encoders()
            self._build_classifier()
            
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.train_op = self.optimizer.apply_gradients(zip(clipped_gradients, params))    
        
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
        self.embed_bi = tf.get_variable("embedding_bi",  dtype=tf.float32, initializer=embed)  
        self.embed_multi = tf.get_variable("embedding_multi",  dtype=tf.float32, initializer=embed)  
        
    def _build_input(self):
        with tf.variable_scope("input"):
            self.text_string = tf.placeholder(tf.string, (None, None, None), "text_string")
            self.len_word = tf.placeholder(tf.int32, (None, None), "len_word")
            self.len_doc = tf.placeholder(tf.int32, (None,), "len_doc")
            
            self.text_bi = tf.nn.embedding_lookup(
                self.embed_bi, self.symbol2index.lookup(self.text_string))
            self.text_multi = tf.nn.embedding_lookup(
                self.embed_multi, self.symbol2index.lookup(self.text_string))
            
            self.x_bi = tf.placeholder(tf.int32, (None,), "x_bi")
            self.y_bi = tf.placeholder(tf.int32, (None,), "y_bi")
            self.relation_bi = tf.placeholder(tf.int32, (None,), "relation_bi")
            self.traditional_features_bi = tf.placeholder(tf.float32, (None, self.dim_traditional), "traditional_features_bi")
            
            self.x_multi = tf.placeholder(tf.int32, (None,), "x_multi")
            self.y_multi = tf.placeholder(tf.int32, (None,), "y_multi")
            self.relation_multi = tf.placeholder(tf.int32, (None,), "relation_multi")
            self.traditional_features_multi = tf.placeholder(tf.float32, (None, self.dim_traditional), "traditional_features_multi")
            
    def _build_encoders(self):
        with tf.variable_scope("encoders"):
            self.enc_text_bi_1 = self._build_encoder(
                tf.reshape(self.text_bi, [-1, tf.shape(self.text_bi)[2], self.dim_embed_word]), 
                tf.reshape(self.len_word, [-1]),
                "encoder_bi_1",
                use_biencoder=True
            )
            self.enc_text_multi_1 = self._build_encoder(
                tf.reshape(self.text_multi, [-1, tf.shape(self.text_multi)[2], self.dim_embed_word]), 
                tf.reshape(self.len_word, [-1]),
                "encoder_multi_1",
                use_biencoder=True
            )      
            self.enc_text_bi_2 = tf.reshape(
                self._build_encoder(
                    tf.reshape(
                        self.enc_text_bi_1, 
                        [tf.shape(self.text_bi)[0], tf.shape(self.text_bi)[1], self.num_units]
                    ),
                    self.len_doc,
                    "encoder_bi_2",
                    use_biencoder=False
                ),
                [tf.shape(self.text_bi)[0] * tf.shape(self.text_bi)[1], self.num_units]
            )                
            self.enc_text_multi_2 = tf.reshape(
                self._build_encoder(
                    tf.reshape(
                        self.enc_text_multi_1, 
                        [tf.shape(self.text_multi)[0], tf.shape(self.text_multi)[1], self.num_units]
                    ),
                    self.len_doc,
                    "encoder_multi_2",
                    use_biencoder=False
                ),
                [tf.shape(self.text_multi)[0] * tf.shape(self.text_multi)[1], self.num_units]
            )            
                
            self.enc_text_bi = self.enc_text_bi_1
            self.enc_text_multi = self.enc_text_multi_1
            self.enc_text_cont_bi = self.enc_text_bi_2
            self.enc_text_cont_multi = self.enc_text_multi_2
            
    def _build_classifier(self):
        with tf.variable_scope("classifier"):
            self.state_bi = tf.concat(
                [
                    tf.matmul(tf.one_hot(self.x_bi, tf.shape(self.enc_text_bi)[0]), self.enc_text_bi),
                    tf.matmul(tf.one_hot(self.y_bi, tf.shape(self.enc_text_bi)[0]), self.enc_text_bi),
                    tf.matmul(tf.one_hot(self.x_bi, tf.shape(self.enc_text_cont_bi)[0]), self.enc_text_cont_bi),
                    tf.matmul(tf.one_hot(self.y_bi, tf.shape(self.enc_text_cont_bi)[0]), self.enc_text_cont_bi),
                    self.traditional_features_bi,
                ],
                axis=-1
            )
            print "state_bi", self.state_bi
            
            self.h_bi = tf.layers.dense(
                self.state_bi,
                self.num_hidden_units,
                activation=tf.tanh
            )
            self.logits_bi = tf.reshape(tf.layers.dense(self.h_bi, 1, activation=tf.sigmoid), [-1])
            self.loss_bi = tf.reduce_mean(-tf.cast(self.relation_bi, tf.float32) * tf.log(self.logits_bi) \
                - (1 - tf.cast(self.relation_bi, tf.float32)) * tf.log(1 - self.logits_bi))
            self.pred_bi = tf.cast(tf.greater(self.logits_bi, 0.5), tf.int32)
            
            self.cnt_pred_bi = tf.reduce_sum(tf.cast(tf.equal(self.pred_bi, 1), tf.float32))
            self.cnt_golden_bi = tf.reduce_sum(tf.cast(tf.equal(self.relation_bi, 1), tf.float32))
            self.cnt_cor_bi = tf.reduce_sum(
                tf.cast(tf.logical_and(tf.equal(self.pred_bi, 1), tf.equal(self.relation_bi, 1)), tf.float32))
                
            self.state_multi = tf.concat(
                [
                    tf.matmul(tf.one_hot(self.x_multi, tf.shape(self.enc_text_multi)[0]), self.enc_text_multi),
                    tf.matmul(tf.one_hot(self.y_multi, tf.shape(self.enc_text_multi)[0]), self.enc_text_multi),
                    tf.matmul(tf.one_hot(self.x_multi, tf.shape(self.enc_text_cont_multi)[0]), self.enc_text_cont_multi),
                    tf.matmul(tf.one_hot(self.y_multi, tf.shape(self.enc_text_cont_multi)[0]), self.enc_text_cont_multi),
                    self.traditional_features_multi,
                ],
                axis=-1
            )   
            print "state_multi", self.state_multi
                  
            self.h_multi = tf.layers.dense(
                self.state_multi,
                self.num_hidden_units,
                activation=tf.tanh
            )
            self.logits_multi = tf.layers.dense(self.h_multi, self.num_relations)
            self.loss_multi = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.relation_multi, logits=self.logits_multi))
            self.pred_multi = tf.cast(tf.argmax(self.logits_multi, axis=-1), tf.int32)
            
            if self.method == "ilp":
                self.weight_bi = self.logits_bi
                self.weight_multi = tf.nn.softmax(self.logits_multi)
            else:
                self.weight = tf.reduce_max(
                    tf.transpose(tf.transpose(tf.nn.softmax(self.logits_multi)) * self.logits_bi)
                    , axis=-1)
                
            self.loss = self.loss_bi + self.loss_multi
            
    def _build_encoder(self, inputs, length, scope, use_biencoder, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            if use_biencoder: 
                cell_fw, cell_bw = self._build_encoder_cell(use_biencoder)
                outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=inputs,
                    sequence_length=length,
                    dtype=tf.float32
                )
                enc_state = []
                for i in range(self.num_layers):
                    enc_state.append(tf.concat([states[0][i], states[1][i]], axis=-1))
                return enc_state[-1]
            else: # high level
                cell = self._build_encoder_cell(use_biencoder)
                outputs, states = tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=inputs,
                    sequence_length=length,
                    dtype=tf.float32
                )
                return outputs
        
    def _build_encoder_cell(self, use_biencoder):
        if use_biencoder:
            cell_fw = tf.contrib.rnn.MultiRNNCell([
                tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.GRUCell(self.num_units / 2), self.keep_prob
                ) for _ in range(self.num_layers)])
            cell_bw = tf.contrib.rnn.MultiRNNCell([
                tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.GRUCell(self.num_units / 2), self.keep_prob
                ) for _ in range(self.num_layers)])
            return cell_fw, cell_bw
        else:
            cell = tf.contrib.rnn.MultiRNNCell([
                tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.GRUCell(self.num_units), self.keep_prob
                ) for _ in range(self.num_layers)])
            return cell
        
    def initialize(self, vocab):
        op_in = self.symbol2index.insert(
            tf.constant(vocab), tf.constant(range(len(vocab)), dtype=tf.int64))
        op_out = self.index2symbol.insert(
            tf.constant(range(len(vocab)), dtype=tf.int64), tf.constant(vocab))
        self.sess.run([op_in, op_out])        

    def step(self, data, train=False):
        input_feed = {
            self.text_string: data["text_string"],
            self.len_word: data["len_word"],
            self.len_doc: data["len_doc"],
            self.x_bi: data["x_bi"],
            self.y_bi: data["y_bi"],
            self.relation_bi: data["relation_bi"],
            self.traditional_features_bi: data["traditional_features_bi"],
            self.x_multi: data["x_multi"],
            self.y_multi: data["y_multi"],
            self.relation_multi: data["relation_multi"],
            self.traditional_features_multi: data["traditional_features_multi"],
        }
        output_feed = [
            self.loss,
            self.loss_bi,
            self.loss_multi,
            self.cnt_pred_bi,
            self.cnt_golden_bi,
            self.cnt_cor_bi,
            self.pred_bi,
            self.pred_multi,
            self.logits_bi,
            self.logits_multi,
        ]
        if train:    
            input_feed[self.keep_prob] = self.train_keep_prob
            output_feed.append(self.train_op)

        ops = self.sess.run(output_feed, input_feed)

        pred_bi = ops[6]
        pred_multi = ops[7]

        cnt_cor_multi = 0

        for i in range(len(data["idx"])):
            j = data["idx"][i]
            if pred_bi[j] == data["relation_bi"][j] and pred_multi[i] == data["relation_multi"][i]:
                cnt_cor_multi += 1
                
        return ops[:6] + [cnt_cor_multi]# + [ops[11]]
        
    def infer(self, data):
        input_feed = {
            self.text_string: data["text_string"],
            self.len_word: data["len_word"],
            self.len_doc: data["len_doc"],
            self.x_bi: data["x_bi"],
            self.y_bi: data["y_bi"],
            self.traditional_features_bi: data["traditional_features_bi"],
            self.x_multi: data["x_multi"],
            self.y_multi: data["y_multi"],
            self.traditional_features_multi: data["traditional_features_multi"],
        }
        if self.method == "ilp":
            output_feed = [
                self.pred_multi,
                self.weight_bi, 
                self.weight_multi
            ]
        else:
            output_feed = [
                self.pred_multi,
                self.logits_bi
            ]
        return self.sess.run(output_feed, input_feed)
