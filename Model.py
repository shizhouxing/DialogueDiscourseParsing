import tensorflow as tf
import numpy as np
import random, copy, math
from Agent import Agent
from NonStructured_Encoder import NonStructured_Encoder
from utils import update_buffer, init_grad

class Model():
    def __init__(self, sess, FLAGS, embed, data_train=None):
        self.sess = sess
        self.num_relations = FLAGS.num_relations
        self.num_units = FLAGS.num_units
        self.dim_embed_relation = FLAGS.dim_embed_relation
        self.max_edu_dist = FLAGS.max_edu_dist
        self.dim_feature_bi = FLAGS.dim_feature_bi
        self.use_structured = FLAGS.use_structured
        self.use_speaker_attn = FLAGS.use_speaker_attn
        self.use_shared_encoders = FLAGS.use_shared_encoders
        self.use_random_structured = FLAGS.use_random_structured
        self.use_traditional = FLAGS.use_traditional
        
        self.learning_rate = tf.Variable(
            float(FLAGS.learning_rate), trainable=False, dtype=tf.float32)                                   
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * FLAGS.learning_rate_decay)
                
        self.agent_bi = Agent(sess, FLAGS, embed, scope="agent_bi", is_multi=False)
        self.agent_multi = Agent(sess, FLAGS, embed, scope="agent_multi", is_multi=True)

        self.params_all = []
        self.params_all += self.agent_bi.params_policy_network
        if self.use_structured:
            self.params_all += self.agent_bi.s_encoder_attn.params
            self.params_all += self.agent_bi.s_encoder_general.params
        self.params_all += self.agent_bi.ns_encoder.params
        self.params_all += self.agent_multi.params_policy_network
        if self.use_structured:
            self.params_all += self.agent_multi.s_encoder_attn.params
            self.params_all += self.agent_multi.s_encoder_general.params
        self.params_all += self.agent_multi.ns_encoder.params
        
        self.grad_unclipped = [
            tf.placeholder(tf.float32, param.shape)
            for param in self.params_all
        ]
        self.grad_clipped = tf.clip_by_global_norm(self.grad_unclipped, 5.0)
        
    def initialize(self, vocab):
        self.agent_bi.ns_encoder.initialize(vocab)
        self.agent_multi.ns_encoder.initialize(vocab)
            
    def sample_action(self, policy):
        action = []
        for p in policy:
            action.append(np.argmax(p))
        return action

    def new_hp_bp_buf(self):
        buf = {
            "grad_bi": [],
            "parent_bi": [],
            "current_bi": [],
            "grad_multi": [],
            "parent_multi": [],
            "current_multi": [],
            "relation": [],
            "idx_parent": [],
            "idx_current": []
        }
        return [copy.deepcopy(buf), copy.deepcopy(buf)]       
        
    def backpropagate_hp(self, batch, k, j):
        if not self.use_structured: return
        speaker_j = batch[k]["edus"][j]["speaker"]
        
        for l in range(self.cnt_speakers[k]):
            if abs(np.sum(self.grad_hp_bi[self.sentence_idx[k][j]][l])) < 1e-9\
                and abs(np.sum(self.grad_hp_multi[self.sentence_idx[k][j]][l])) < 1e-9 : continue
            attn = bool(l == speaker_j)              
            if not self.use_speaker_attn:
                attn = 0  
            for _i in range(len(self.parents_hp[k][j])):
                par = self.parents[k][j][_i]
                self.hp_bp_buf[attn]["grad_bi"].append(np.array(self.grad_hp_bi[self.sentence_idx[k][j]][l]))
                self.hp_bp_buf[attn]["parent_bi"].append(np.array(self.hp_bi[self.sentence_idx[k][par]][l]))
                self.hp_bp_buf[attn]["current_bi"].append(self.hs_bi[self.sentence_idx[k][j]])
                self.hp_bp_buf[attn]["grad_multi"].append(np.array(self.grad_hp_multi[self.sentence_idx[k][j]][l]))
                self.hp_bp_buf[attn]["parent_multi"].append(np.array(self.hp_multi[self.sentence_idx[k][par]][l]))
                self.hp_bp_buf[attn]["current_multi"].append(self.hs_multi[self.sentence_idx[k][j]])
                self.hp_bp_buf[attn]["relation"].append(self.parents_relation_hp[k][j][_i])
                self.hp_bp_buf[attn]["idx_parent"].append((self.sentence_idx[k][par], l))
                self.hp_bp_buf[attn]["idx_current"].append(self.sentence_idx[k][j])

            self.hp_bp_buf[attn]["grad_bi"].append(np.array(self.grad_hp_bi[self.sentence_idx[k][j]][l]))
            self.hp_bp_buf[attn]["parent_bi"].append(self.zero)
            self.hp_bp_buf[attn]["current_bi"].append(self.hs_bi[self.sentence_idx[k][j]])
            self.hp_bp_buf[attn]["grad_multi"].append(np.array(self.grad_hp_multi[self.sentence_idx[k][j]][l]))
            self.hp_bp_buf[attn]["parent_multi"].append(self.zero)
            self.hp_bp_buf[attn]["current_multi"].append(self.hs_multi[self.sentence_idx[k][j]])
            self.hp_bp_buf[attn]["relation"].append(self.num_relations)
            self.hp_bp_buf[attn]["idx_parent"].append(None)
            self.hp_bp_buf[attn]["idx_current"].append(self.sentence_idx[k][j])

            self.grad_hp_bi[self.sentence_idx[k][j]][l] = np.zeros(self.num_units)            
            self.grad_hp_multi[self.sentence_idx[k][j]][l] = np.zeros(self.num_units)            
               
    def update_gradients(self, g1, g2):
        if g2 is None: return
        if g1 is None:
            return np.array(g2)
        else:
            for l in range(len(g1)):
                g1[l] += g2[l]
            return g1
       
    def get_sum(self, grad):
        s = 0
        for item in grad:
            s += np.sum(item)
        return s
                
    def backpropagate_hp_flush(self):
        if not self.use_structured: return
        o_feed, i_feed = [], {}
        for attn in range(0, 2):
            if len(self.hp_bp_buf[attn]["idx_parent"]) == 0: continue
            def update_gradients_buffer(o_feed, i_feed, agent, name):
                return update_buffer(
                    o_feed, i_feed,
                    (agent.s_encoder_attn if attn else agent.s_encoder_general).get_gradients(
                        self.hp_bp_buf[attn]["grad_%s" % name],
                        self.hp_bp_buf[attn]["parent_%s" % name],
                        self.hp_bp_buf[attn]["current_%s" % name],
                        self.hp_bp_buf[attn]["relation"],
                        buffered=True
                    )
                )
            o_feed, i_feed = update_gradients_buffer(o_feed, i_feed, self.agent_bi, "bi")
            o_feed, i_feed = update_gradients_buffer(o_feed, i_feed, self.agent_multi, "multi")
            
        res = self.sess.run(o_feed, i_feed)
        c = 0
   
        for attn in range(0, 2):
            if len(self.hp_bp_buf[attn]["idx_parent"]) == 0: continue
            def update_gradients(agent, grad_hp, grad_hs, g_structured, g_parent, g_current):
                for k, idx in enumerate(self.hp_bp_buf[attn]["idx_parent"]):
                    if idx is not None:
                        grad_hp[idx[0]][idx[1]] += g_parent[k]
                for k, idx in enumerate(self.hp_bp_buf[attn]["idx_current"]):
                    grad_hs[idx] += g_current[k]
                  
                if attn:
                    agent.grad_s_encoder_attn = self.update_gradients(agent.grad_s_encoder_attn, g_structured)
                else:
                    agent.grad_s_encoder_general = self.update_gradients(agent.grad_s_encoder_general, g_structured)
            
            update_gradients(
                self.agent_bi, self.grad_hp_bi, self.grad_hs_bi, 
                res[c], res[c + 1][0], res[c + 1][1]
            )
            c += 2
            update_gradients(
                self.agent_multi, self.grad_hp_multi, self.grad_hs_multi, 
                res[c], res[c + 1][0], res[c + 1][1]
            )   
            c += 2
                
        self.hp_bp_buf = self.new_hp_bp_buf()        
        
    def backpropagate_hp_all(self, batch):
        if self.use_structured:
            # hp backpropagation
            for k, dialog in enumerate(batch):
                for j in range(len(dialog["edus"]) - 1, -1, -1):
                    self.backpropagate_hp(batch, k, j)
            self.backpropagate_hp_flush()

    def get_hs(self, batch):
        self.max_num_edus = max([len(dialog["edus"]) for dialog in batch])
        self.edus, self.num_posts = [], []
        for dialog in batch:
            self.edus.append([])
            for edu in dialog["edus"]:
                self.edus[-1].append(edu["tokens"])
            for i in range(self.max_num_edus - len(dialog["edus"])):
                self.edus[-1].append([])
            self.num_posts.append(len(dialog["edus"]))
        
        o_feed, i_feed = [], {}
        o_feed, i_feed = update_buffer(
            o_feed, i_feed, 
            self.agent_bi.ns_encoder.infer(self.edus, self.num_posts, is_train=self.is_train, buffered=True)
        )
        o_feed, i_feed = update_buffer(
            o_feed, i_feed, 
            self.agent_multi.ns_encoder.infer(self.edus, self.num_posts, is_train=self.is_train, buffered=True)
        )

        res = self.sess.run(o_feed, i_feed)
                
        self.sentences = []
        self.sentence_idx = []
        for dialog in batch:
            idx = []
            for edu in dialog["edus"]:
                self.sentences.append(edu["tokens"])
                idx.append(len(self.sentences) - 1)
            self.sentence_idx.append(idx)
            
        self.hs_bi, self.hs_multi, self.hs_idp, self.hc_bi, self.hc_multi = [], [], [], [], []
        for i, dialog in enumerate(batch):
            for j in range(len(dialog["edus"])):
                idx = i * self.max_num_edus + j
                self.hs_bi.append(res[0][idx])
                self.hs_multi.append(res[3][idx])
                self.hc_bi.append(res[1][idx])
                self.hc_multi.append(res[4][idx])
        
        self.agent_bi.ns_encoder.recurrent_noise = res[2]
        self.agent_multi.ns_encoder.recurrent_noise = res[5]
            
        self.hs_bi = np.array(self.hs_bi)
        self.hs_multi = np.array(self.hs_multi)
        self.grad_hs_bi = np.zeros(self.hs_bi.shape)      
        self.grad_hs_multi = np.zeros(self.hs_multi.shape)  
        self.hc_bi = np.array(self.hc_bi)
        self.hc_multi = np.array(self.hc_multi)
        self.grad_hc_bi = np.zeros(self.hc_bi.shape)      
        self.grad_hc_multi = np.zeros(self.hc_multi.shape)  
    
    def count_speakers(self, batch):
        self.cnt_speakers = []
        for i, dialog in enumerate(batch):
            speakers = {}
            for edu in dialog["edus"]:
                if not edu["speaker"] in speakers:
                    speakers[edu["speaker"]] = len(speakers)
                edu["speaker"] = speakers[edu["speaker"]]
            self.cnt_speakers.append(len(speakers))
            
    def get_hp_new_buf(self):
        buf = {
            "parent_bi": [],
            "current_bi": [],
            "parent_multi": [],
            "current_multi": [],
            "relation": [],
            "idx": []
        }
        return [copy.deepcopy(buf), copy.deepcopy(buf)] 
            
    def init_hp(self, batch):
        # parent path representation
        self.hp_bi = np.zeros((len(self.sentences), max(self.cnt_speakers), self.num_units))
        self.hp_multi = np.zeros((len(self.sentences), max(self.cnt_speakers), self.num_units))
        
        self.cntp = np.ones(len(self.sentences))    
        self.is_root = np.ones(len(self.sentences)) 
           
        # root
        self.hp_new_buf = self.get_hp_new_buf()
        self.zero = np.zeros(self.num_units)
        
        for k, dialog in enumerate(batch):
            for j in range(len(dialog["edus"])):
                idx_j = self.sentence_idx[k][j] 
                for l in range(self.cnt_speakers[k]):
                    attn = bool(l == batch[k]["edus"][j]["speaker"])
                    if not self.use_speaker_attn:
                        attn = 0
                    self.hp_new_buf[attn]["parent_bi"].append(self.zero)
                    self.hp_new_buf[attn]["current_bi"].append(self.hs_bi[idx_j])
                    self.hp_new_buf[attn]["parent_multi"].append(self.zero)
                    self.hp_new_buf[attn]["current_multi"].append(self.hs_multi[idx_j])
                    self.hp_new_buf[attn]["relation"].append(self.num_relations)
                    self.hp_new_buf[attn]["idx"].append((idx_j, l)) 
        self.update_hp(batch, fixed_noise=0)
        self.hp_bp_buf = self.new_hp_bp_buf()

        self.grad_hp_bi = np.zeros(self.hp_bi.shape)
        self.grad_hp_multi = np.zeros(self.hp_multi.shape)
        
    def build_relation_list(self, batch):
        # relation list
        cnt_golden = 0
        self.relation_list = []
        self.relation_types = []
        self.parents = []
        self.parents_relation = []
        self.parents_hp = []
        self.parents_relation_hp = []
        for k, dialog in enumerate(batch):
            self.parents.append([[] for i in range(len(dialog["edus"]))])
            self.parents_relation.append([[] for i in range(len(dialog["edus"]))])
            self.parents_hp.append([[] for i in range(len(dialog["edus"]))])
            self.parents_relation_hp.append([[] for i in range(len(dialog["edus"]))])
            self.relation_types.append(np.zeros((len(dialog["edus"]), len(dialog["edus"])), dtype=np.int32))
            for relation in dialog["relations"]:
                self.relation_types[k][relation["x"]][relation["y"]] = relation["type"] + 1
                cnt_golden += 1
            for j in range(len(dialog["edus"])):
                r = []
                for i in range(len(dialog["edus"])):
                    if self.relation_types[k][i][j] > 0 and \
                        (i < j and j - i <= self.max_edu_dist):
                            r.append(i)
                self.relation_list.append(r)        
        return cnt_golden
        
    def get_state(self, batch, hs, hc, hp, k, i, j):
        idx_i = self.sentence_idx[k][i]
        idx_j = self.sentence_idx[k][j]
        speaker_i = batch[k]["edus"][i]["speaker"]
        speaker_j = batch[k]["edus"][j]["speaker"]
        
        h = np.concatenate([
            hc[idx_i],
            hs[idx_j],
        ], axis=-1)      
        if self.use_structured:
            h = np.concatenate([
                h,
                hp[idx_i][speaker_j],
                hc[idx_j],
            ], axis=-1)
        else:
            h = np.concatenate([
                h,
                hs[idx_i],
                hc[idx_j],
            ], axis=-1)
            
        if self.use_traditional:
            h = np.concatenate([
                h,
                [
                    j - i, 
                    speaker_i == speaker_j,
                    batch[k]["edus"][i]["turn"] == batch[k]["edus"][j]["turn"],
                    (i in self.parents[k][j]) or (j in self.parents[k][i])
                ]
            ], axis=-1)
            
        return h
        
    def update_grad_state(self, batch, grad_hs, grad_hc, grad_hp, g_state, k, i, j):
        idx_i = self.sentence_idx[k][i]
        idx_j = self.sentence_idx[k][j]
        speaker_i = batch[k]["edus"][i]["speaker"]
        speaker_j = batch[k]["edus"][j]["speaker"]   
        
        grad_hc[idx_i] += g_state[0:self.num_units]
        grad_hs[idx_j] += g_state[self.num_units:2*self.num_units]
        if self.use_structured:
            grad_hp[idx_i][speaker_j] += g_state[2*self.num_units:3*self.num_units]
            grad_hc[idx_j] += g_state[3*self.num_units:4*self.num_units]
        else:
            grad_hs[idx_i] += g_state[2*self.num_units:3*self.num_units]
            grad_hc[idx_j] += g_state[3*self.num_units:4*self.num_units]         
            
    def new_edge(self, batch, k, i, j, r):
        # bp gradients of hp first before a new parent is added
        if self.use_structured:
            self.backpropagate_hp(batch, k, j)
        
        self.parents[k][j].append(i)
        self.parents_relation[k][j].append(r)
        
        if self.use_random_structured:
            i = random.randint(0, j - 1)
            r = random.randint(0, self.num_relations - 1)
        
        self.parents_hp[k][j].append(i)
        self.parents_relation_hp[k][j].append(r)
        
        idx_j = self.sentence_idx[k][j]
        if self.use_structured:
            if self.is_root[idx_j]:
                self.is_root[idx_j] = 0
                self.cntp[idx_j] = 1
            else:
                self.cntp[idx_j] += 1
                
            for l in range(self.cnt_speakers[k]):
                attn = bool(l == batch[k]["edus"][j]["speaker"])
                if not self.use_speaker_attn:
                    attn = 0
                self.hp_new_buf[attn]["parent_bi"].append(np.array(self.hp_bi[self.sentence_idx[k][i]][l]))
                self.hp_new_buf[attn]["current_bi"].append(self.hs_bi[idx_j])
                self.hp_new_buf[attn]["parent_multi"].append(np.array(self.hp_multi[self.sentence_idx[k][i]][l]))
                self.hp_new_buf[attn]["current_multi"].append(self.hs_multi[idx_j])
                self.hp_new_buf[attn]["relation"].append(r)
                self.hp_new_buf[attn]["idx"].append((idx_j, l))   
           
    def update_hp(self, batch, fixed_noise=1):    
        o_feed, i_feed = [], {}
        
        for attn in range(0, 2):
            if len(self.hp_new_buf[attn]["idx"]) == 0: continue
            def update_hp_buffer(o_feed, i_feed, agent, name, hp):
                self.hp_new_buf[attn]["parent"] = np.array(self.hp_new_buf[attn]["parent_%s" % name])
                self.hp_new_buf[attn]["current"] = np.array(self.hp_new_buf[attn]["current_%s" % name])
                self.hp_new_buf[attn]["relation"] = np.array(self.hp_new_buf[attn]["relation"])
                return update_buffer(
                    o_feed, i_feed,
                    (agent.s_encoder_attn if attn else agent.s_encoder_general)\
                        .infer(self.hp_new_buf[attn], fixed_noise, buffered=True)
                )
            o_feed, i_feed = update_hp_buffer(o_feed, i_feed, self.agent_bi, "bi", self.hp_bi)
            o_feed, i_feed = update_hp_buffer(o_feed, i_feed, self.agent_multi, "multi", self.hp_multi)
                
        res = self.sess.run(o_feed, i_feed)
        c = 0
        
        for attn in range(0, 2):
            if len(self.hp_new_buf[attn]["idx"]) == 0: continue
            def update_hp(hp, _hp):
                for i, idx in enumerate(self.hp_new_buf[attn]["idx"]):
                    if self.cntp[idx[0]] == 1:
                        hp[idx[0]][idx[1]] = _hp[i]
                    else:
                        hp[idx[0]][idx[1]] += _hp[i]
            update_hp(self.hp_bi, res[c])
            if attn:
                self.agent_bi.s_encoder_attn.recurrent_noise = res[c + 1]
            else:
                self.agent_bi.s_encoder_general.recurrent_noise = res[c + 1]
            c += 2

            update_hp(self.hp_multi, res[c])
            if attn:
                self.agent_multi.s_encoder_attn.recurrent_noise = res[c + 1]
            else:
                self.agent_multi.s_encoder_general.recurrent_noise = res[c + 1]
            c += 2
         
    def clip_gradients(self):
        gradients = []
        def append_grad(grad):            
            return gradients + list(grad)
        gradients = append_grad(self.agent_bi.grad_policy)
        if self.use_structured:
            gradients = append_grad(self.agent_bi.grad_s_encoder_attn)
            gradients = append_grad(self.agent_bi.grad_s_encoder_general)
        gradients = append_grad(self.agent_bi.grad_ns_encoder)
        gradients = append_grad(self.agent_multi.grad_policy)
        if self.use_structured:
            gradients = append_grad(self.agent_multi.grad_s_encoder_attn)
            gradients = append_grad(self.agent_multi.grad_s_encoder_general)
        gradients = append_grad(self.agent_multi.grad_ns_encoder)
        
        input_feed = {}
        for i in range(len(gradients)):
            input_feed[self.grad_unclipped[i]] = gradients[i]
        gradients_clipped, global_norm = self.sess.run(
            self.grad_clipped, input_feed)
            
        def get_clipped(grad):
            return gradients_clipped[:len(grad)], gradients_clipped[len(grad):]
            
        self.agent_bi.grad_policy, gradients_clipped = get_clipped(self.agent_bi.grad_policy)
        if self.use_structured:
            self.agent_bi.grad_s_encoder_attn, gradients_clipped = get_clipped(self.agent_bi.grad_s_encoder_attn)
            self.agent_bi.grad_s_encoder_general, gradients_clipped = get_clipped(self.agent_bi.grad_s_encoder_general)
        self.agent_bi.grad_ns_encoder, gradients_clipped = get_clipped(self.agent_bi.grad_ns_encoder)
        self.agent_multi.grad_policy, gradients_clipped = get_clipped(self.agent_multi.grad_policy)
        if self.use_structured:
            self.agent_multi.grad_s_encoder_attn, gradients_clipped = get_clipped(self.agent_multi.grad_s_encoder_attn)
            self.agent_multi.grad_s_encoder_general, gradients_clipped = get_clipped(self.agent_multi.grad_s_encoder_general)
        self.agent_multi.grad_ns_encoder, gradients_clipped = get_clipped(self.agent_multi.grad_ns_encoder)
         
    def train(self, batch):
        grad_hs_bi = np.zeros((len(batch) * self.max_num_edus, self.hs_bi.shape[1]))
        grad_hc_bi = np.zeros((len(batch) * self.max_num_edus, self.hc_bi.shape[1]))
        grad_hs_multi = np.zeros((len(batch) * self.max_num_edus, self.hs_multi.shape[1]))
        grad_hc_multi = np.zeros((len(batch) * self.max_num_edus, self.hc_multi.shape[1]))
        
        cur = 0
        for i, dialog in enumerate(batch):
            grad_hs_bi[i * self.max_num_edus: i * self.max_num_edus + len(dialog["edus"]), :] = \
                self.grad_hs_bi[cur : cur + len(dialog["edus"]), :]
            grad_hs_multi[i * self.max_num_edus: i * self.max_num_edus + len(dialog["edus"]), :] = \
                self.grad_hs_multi[cur : cur + len(dialog["edus"]), :]
            grad_hc_bi[i * self.max_num_edus: i * self.max_num_edus + len(dialog["edus"]), :] = \
                self.grad_hc_bi[cur : cur + len(dialog["edus"]), :]
            grad_hc_multi[i * self.max_num_edus: i * self.max_num_edus + len(dialog["edus"]), :] = \
                self.grad_hc_multi[cur : cur + len(dialog["edus"]), :]
            cur += len(dialog["edus"])
        
        o_feed, i_feed = self.agent_bi.ns_encoder\
            .get_gradients(self.edus, self.num_posts, grad_hs_bi, grad_hc_bi, buffered=True)
        o_feed, i_feed = update_buffer(
            o_feed, i_feed, 
            self.agent_multi.ns_encoder.get_gradients(
                self.edus, self.num_posts, grad_hs_multi, grad_hc_multi, buffered=True)
        )
        res = self.sess.run(o_feed, i_feed)
        self.agent_bi.grad_ns_encoder = res[0]
        self.agent_multi.grad_ns_encoder = res[1]
            
        self.clip_gradients()
                    
        output_feed, input_feed = self.agent_bi.train(self.learning_rate.eval(), buffered=True)
        output_feed, input_feed = update_buffer(
            output_feed, input_feed, 
            self.agent_multi.train(self.learning_rate.eval(), buffered=True)
        )
        
        self.sess.run(output_feed, input_feed)        
                
    def step(self, batch, is_train=False):
        self.is_train = is_train
        
        self.agent_bi.clear_gradients()
        self.agent_multi.clear_gradients()
        
        cnt_golden, cnt_pred, cnt_cor_bi, cnt_cor_multi = 0, 0, 0, 0
        sum_loss_bi, cnt_loss_bi = 0, 0
        sum_loss_multi, cnt_loss_multi = 0, 0
        
        self.get_hs(batch)
        self.count_speakers(batch)
        if self.use_structured: 
            self.init_hp(batch)
        else:
            self.hp_bi, self.hp_multi = None, None
            self.grad_hp_bi, self.grad_hp_multi = None, None
        cnt_golden = self.build_relation_list(batch)
        
        cur = [(1, 0)] * len(batch)
        unfinished = np.ones(len(batch), dtype=np.int32)
        max_edus = max([len(dialog["edus"]) for dialog in batch])
        for k, dialog in enumerate(batch):
            if len(dialog["edus"]) <= 1:
                unfinished[k] = False
        
        while (np.sum(unfinished) > 0):
            size = np.sum(unfinished)
            state = np.zeros((size, max_edus + 1, self.agent_bi.dim_state))
            state_multi = np.zeros((size, max_edus + 1, self.agent_multi.dim_state))
            mask = np.zeros((size, max_edus + 1))
            golden = np.zeros(size, dtype=np.int32)
            idx = 0
            
            for k, dialog in enumerate(batch):
                if not unfinished[k]: continue
                j = cur[k][0]
                idx_j = self.sentence_idx[k][j]
                for i in range(j):
                    if j - i <= self.max_edu_dist:
                        if (i in self.parents[k][j]): continue
                        idx_i = self.sentence_idx[k][i]
                        state[idx][i] = self.get_state(
                            batch, 
                            self.hs_bi,
                            self.hc_bi,
                            self.hp_bi, 
                            k, i, j
                        )
                        state_multi[idx][i] = self.get_state(
                            batch, 
                            self.hs_multi,
                            self.hc_multi,
                            self.hp_multi, 
                            k, i, j
                        )                        
                        mask[idx][i] = 1
                    
                golden[idx] = 0
                for i in self.relation_list[idx_j]:
                    if (i in self.parents[k][j]): continue
                    golden[idx] = i
                    break
                idx += 1
                
            # sample an action
            policy = self.agent_bi.get_policy(state, mask)
                
            action = self.sample_action(policy)
            if not action:
                print policy
                raise Warning("Action not found, policy:")
                
            # update prec/recall statistics
            idx = 0
            for k, dialog in enumerate(batch):
                if not unfinished[k]: continue
                # predicted a new relation
                if action[idx] != len(dialog["edus"]):
                    cnt_pred += 1
                    if self.relation_types[k][action[idx]][cur[k][0]] > 0:
                        cnt_cor_bi += 1
                idx += 1
                
            if is_train:
                # use MLE loss (bi)
                loss, g_policy, g_state = self.agent_bi.get_gradients(state, golden, mask)                
                
                # accumulate gradient for policy network
                self.agent_bi.grad_policy = self.update_gradients(
                    self.agent_bi.grad_policy, g_policy)
                     
                # accumulate gradient for hs and hp
                idx = 0
                for k, dialog in enumerate(batch):
                    if not unfinished[k]: continue                    
                    j = cur[k][0]
                    idx_j = self.sentence_idx[k][j]
                    speaker_j = dialog["edus"][j]["speaker"]
                    for i in range(len(dialog["edus"])):
                        if mask[idx][i] > 0:
                            self.update_grad_state(
                                batch, self.grad_hs_bi, self.grad_hc_bi, self.grad_hp_bi, g_state[idx, i, :], k, i, j) 
                    idx += 1
                        
                sum_loss_bi += loss
                cnt_loss_bi += 1
                    
            # predict labels        
            idx = 0
            state_multi, golden_multi, idx_multi = [], [], []
            state_multi_train, golden_multi_train, idx_multi_train = [], [], []
            for k, dialog in enumerate(batch):
                if not unfinished[k]: continue
                j = cur[k][0]
                if action[idx] != len(dialog["edus"]):
                    i = action[idx]
                    if self.use_shared_encoders:
                        state_multi.append(self.get_state(
                            batch, 
                            self.hs_bi,
                            self.hc_bi,
                            self.hp_bi, 
                            k, i, j
                        ))
                    else:
                        state_multi.append(self.get_state(
                            batch, 
                            self.hs_multi,
                            self.hc_multi,
                            self.hp_multi, 
                            k, i, j
                        ))
                    idx_multi.append((k, i, j))   
                for i in range(j):
                    if self.relation_types[k][i][j] > 0:
                        if i in self.parents[k][j]: continue
                        if self.use_shared_encoders:
                            state_multi_train.append(self.get_state(
                                batch, 
                                self.hs_bi,
                                self.hc_bi,
                                self.hp_bi, 
                                k, i, j)
                            )
                        else:
                            state_multi_train.append(self.get_state(
                                batch, 
                                self.hs_multi,
                                self.hc_multi,
                                self.hp_multi, 
                                k, i, j)
                            )
                        idx_multi_train.append((k, i, j))
                        golden_multi_train.append(self.relation_types[k][i][j] - 1)
                idx += 1
            if len(idx_multi) > 0:
                policy = self.agent_multi.get_policy(state_multi)
                labels = self.sample_action(policy)
                
            # use MLE loss (multi)
            if len(idx_multi_train) > 0:
                loss, g_policy, g_state = self.agent_multi.get_gradients(state_multi_train, golden_multi_train)
                
                if is_train:
                    # accumulate gradient for policy network
                    self.agent_multi.grad_policy = self.update_gradients(
                        self.agent_multi.grad_policy, g_policy)
                         
                    # accumulate gradient for self.hs and self.hp
                    for l, idx in enumerate(idx_multi_train):
                        k, i, j = idx[0], idx[1], idx[2]
                        if self.use_shared_encoders:
                            self.update_grad_state(
                                batch, self.grad_hs_bi, self.grad_hc_bi, self.grad_hp_bi, g_state[l, :], k, i, j)  
                        else:
                            self.update_grad_state(
                                batch, self.grad_hs_multi, self.grad_hc_multi, self.grad_hp_multi, g_state[l, :], k, i, j)  
                sum_loss_multi += loss
                cnt_loss_multi += 1    
                    
            # update prec/recall statistics
            idx, idx_multi = 0, 0
            for k, dialog in enumerate(batch):
                if not unfinished[k]: continue    
                # predicted a new relation
                if action[idx] != len(dialog["edus"]):
                    if labels[idx_multi] == self.relation_types[k][action[idx]][cur[k][0]] - 1:
                        cnt_cor_multi += 1
                    idx_multi += 1
                idx += 1
                  
            # buffer for updating parent path representations   
            self.hp_new_buf = self.get_hp_new_buf()
                           
            # take action   
            if self.use_structured:
                self.hp_bp_buf = self.new_hp_bp_buf()
            idx, idx_multi, idx_multi_train = 0, 0, 0
            for k, dialog in enumerate(batch):
                if not unfinished[k]: continue
                # valid prediction
                if action[idx] != len(dialog["edus"]):
                    r = labels[idx_multi]
                    if self.relation_types[k][action[idx]][cur[k][0]] > 0:
                        idx_multi_train += 1
                    idx_multi += 1
                    self.new_edge(batch, k, action[idx], cur[k][0], r)
                cur[k] = (cur[k][0] + 1, 0)
                if cur[k][0] >= len(dialog["edus"]):
                    unfinished[k] = False                    
                idx += 1
            if self.use_structured:
                self.backpropagate_hp_flush()
                self.update_hp(batch)
                    
        # update the parameters        
        if is_train:
            self.backpropagate_hp_all(batch)
            self.train(batch)
                
        relations_pred = []
        for k, dialog in enumerate(batch):
            relations_pred.append([])
            for i in range(len(dialog["edus"])):
                for j in range(len(self.parents[k][i])):
                    relations_pred[k].append((self.parents[k][i][j], i, self.parents_relation[k][i][j]))
            
        if is_train:
            if math.isnan(sum_loss_bi) or math.isnan(sum_loss_multi):
                print "sum_loss_bi", sum_loss_bi
                print "sum_loss_multi", sum_loss_multi
                raise Warning("NaN appears!")
        
        for dialog in batch:
            cnt = [0] * len(dialog["edus"])
            for r in dialog["relations"]:
                cnt[r["y"]] += 1
            for i in range(len(dialog["edus"])):
                if cnt[i] == 0:
                    cnt_golden += 1
            cnt_pred += 1
            if cnt[0] == 0:
                cnt_cor_bi += 1
                cnt_cor_multi += 1
            
        return [
            sum_loss_bi / cnt_loss_bi if cnt_loss_bi > 0 else 0, 
            sum_loss_multi / cnt_loss_multi if cnt_loss_multi > 0 else 0,
            cnt_golden, cnt_pred, cnt_cor_bi, cnt_cor_multi,
            relations_pred,
        ]
