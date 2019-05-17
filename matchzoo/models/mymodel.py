# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding, Dot
from keras.optimizers import Adam
from model import BasicModel
from utils.utility import *
from layers.Match import *

from keras.layers.core import Dense, RepeatVector, Masking, Dropout
from keras.layers.merge import Concatenate
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import GRU, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D

from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import VarianceScaling
from keras.regularizers import *
import tensorflow as tf


class CrossATT(Layer):

    def __init__(self, output_dim, c_maxlen, q_maxlen, dropout, **kwargs):
        self.output_dim=output_dim
        self.c_maxlen = c_maxlen
        self.q_maxlen = q_maxlen
        self.dropout = dropout
        super(CrossATT, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: [(None, ?, 128), (None, ?, 128)]
        init = VarianceScaling(scale=1.0, mode='fan_in', distribution='normal')
        self.W0 = self.add_weight(name='W0',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer=init,
                                  regularizer=l2(3e-7),
                                  trainable=True)
        self.W1 = self.add_weight(name='W1',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer=init,
                                  regularizer=l2(3e-7),
                                  trainable=True)

        super(CrossATT, self).build(input_shape)

    def mask_logits(self, inputs, mask, mask_value = -1e30):
        mask = tf.cast(mask, tf.float32)
        return inputs + mask_value * (1 - mask)

    def call(self, x, mask=None):
        x_cont, x_ques, c_mask, q_mask = x
        S = K.batch_dot(x_cont, K.permute_dimensions(x_ques, pattern=(0, 2, 1)))
        S_ = tf.nn.softmax(S)
        S_n = tf.expand_dims(S_, 3)
        vs = K.tile(S_n, [1, 1, 1, self.output_dim])
        v0 = tf.expand_dims(x_ques, 1)
        v1 = K.tile(v0, [1, self.c_maxlen, 1, 1])
        c2q = tf.multiply(vs, v1)
        v11 = K.sum(c2q, axis=2)
        v2 = K.dot(v11, self.W1)
        v3 = K.dot(x_cont, self.W0)
        result = v2 + v3

        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

class context2query_attention(Layer):

    def __init__(self, output_dim, c_maxlen, q_maxlen, dropout, **kwargs):
        self.output_dim=output_dim
        self.c_maxlen = c_maxlen
        self.q_maxlen = q_maxlen
        self.dropout = dropout
        super(context2query_attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: [(None, ?, 128), (None, ?, 128)]
        init = VarianceScaling(scale=1.0, mode='fan_in', distribution='normal')
        self.W0 = self.add_weight(name='W0',
                                  shape=(input_shape[0][-1], 1),
                                  initializer=init,
                                  regularizer=l2(3e-7),
                                  trainable=True)
        self.W1 = self.add_weight(name='W1',
                                  shape=(input_shape[1][-1], 1),
                                  initializer=init,
                                  regularizer=l2(3e-7),
                                  trainable=True)
        self.W2 = self.add_weight(name='W2',
                                  shape=(1, 1, input_shape[0][-1]),
                                  initializer=init,
                                  regularizer=l2(3e-7),
                                  trainable=True)
        self.bias = self.add_weight(name='linear_bias',
                                    shape=([1]),
                                    initializer='zero',
                                    regularizer=l2(3e-7),
                                    trainable=True)
        super(context2query_attention, self).build(input_shape)

    def mask_logits(self, inputs, mask, mask_value = -1e30):
        mask = tf.cast(mask, tf.float32)
        return inputs + mask_value * (1 - mask)

    def call(self, x, mask=None):
        x_cont, x_ques, c_mask, q_mask = x

        # get similarity matrix S
        subres0 = K.tile(K.dot(x_cont, self.W0), [1, 1, self.q_maxlen])
        subres1 = K.tile(K.permute_dimensions(K.dot(x_ques, self.W1), pattern=(0, 2, 1)), [1, self.c_maxlen, 1])
        subres2 = K.batch_dot(x_cont * self.W2, K.permute_dimensions(x_ques, pattern=(0, 2, 1)))
        S = subres0 + subres1 + subres2
        S += self.bias
        q_mask = tf.expand_dims(q_mask, 1)
        S_ = tf.nn.softmax(self.mask_logits(S, q_mask))
        c_mask = tf.expand_dims(c_mask, 2)
        S_T = K.permute_dimensions(tf.nn.softmax(self.mask_logits(S, c_mask), axis=1), (0, 2, 1))
        c2q = tf.matmul(S_, x_ques)
        q2c = tf.matmul(tf.matmul(S_, S_T), x_cont)
        result = K.concatenate([x_cont, c2q, x_cont * c2q, x_cont * q2c], axis=-1)

        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        # assert len(input_shape) == 3
        print("input_shape", input_shape)
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        # print("x", x.get_shape().as_list())
        # print("W", self.W.get_shape().as_list())
        # print("b", self.b.get_shape().as_list())
        # print("u", self.u.get_shape().as_list())
        # # v1 = tf.matmul(x, self.W)
        # W = K.variable(self.init((x.get_shape().as_list()[-1], self.attention_dim)))
        # print("W1", W.get_shape().as_list())
        v1 = K.dot(x, self.W)
        v2 = K.bias_add(v1, self.b)
        uit = K.tanh(v2)
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        # 自然对数为底的指数
        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        # print("ait", ait.get_shape().as_list())
        # ait是概率List
        # weighted_input = x * ait
        # output = K.sum(weighted_input, axis=1)

        return ait

    def compute_output_shape(self, input_shape):
        # return (input_shape[0], input_shape[-1])
        return (input_shape[0], input_shape[1], 1)

class TileLayer(Layer):
    def __init__(self, dim):
        self.dim = dim
        super(TileLayer, self).__init__()

    def call(self, q_embed, mask=None):
        q_emb_exp = K.expand_dims(q_embed, axis=1)
        show_layer_info('exp 1', q_emb_exp)
        q_emb_reshape = K.tile(q_emb_exp, (1, self.dim, 1))
        show_layer_info('tile 1', q_emb_reshape)
        return q_emb_reshape
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim, input_shape[1])


class SqueezeLayer(Layer):
    def __init__(self, dim):
        # self.init = initializers.get('normal')
        # self.supports_masking = True
        self.dim = dim
        super(SqueezeLayer, self).__init__()

    def call(self, q_embed, mask=None):
        q_emb_exp = K.squeeze(q_embed, axis=self.dim)
        # q_emb_exp = K.squeeze(q_embed)
        show_layer_info('squeeze 1', q_emb_exp)
        return q_emb_exp

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2], input_shape[3])

class MYMODEL(BasicModel):
    def __init__(self, config):
        super(MYMODEL, self).__init__(config)
        self.__name = 'MYMODEL'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'embed', 'embed_size', 'train_embed',  'vocab_size',
                   'hidden_size', 'topk', 'dropout_rate']
        self.embed_trainable = config['train_embed']

        self.setup(config)
        if not self.check():
            raise TypeError('[MYMODEL] parameter check wrong')
        self.sent_num = int(self.config['text2_maxlen']/self.config['text1_maxlen'])
        print('[MYMODEL] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('hidden_size', 32)
        self.set_default('topk', 100)
        self.set_default('dropout_rate', 0)
        self.config.update(config)

    def build(self):

        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)
        sent = Input(shape=(self.config['text1_maxlen'],))

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)
        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        # d_embed = embedding(doc)
        # show_layer_info('Embedding', d_embed)
        s_embed = embedding(sent)
        show_layer_info('Embedding', s_embed)

        q_rep = Bidirectional(LSTM(self.config['hidden_size'], return_sequences=True, dropout=self.config['dropout_rate']))(q_embed)
        show_layer_info('Bidirectional-LSTM', q_rep)
        # d_rep = Bidirectional(GRU(self.config['hidden_size'], return_sequences=True, dropout=self.config['dropout_rate']))(d_embed)
        # show_layer_info('Bidirectional-LSTM', d_rep)
        s_rep = Bidirectional(LSTM(self.config['hidden_size'], return_sequences=True, dropout=self.config['dropout_rate']))(s_embed)
        show_layer_info('Bidirectional-LSTM', s_rep)


        c_mask = Lambda(lambda x: tf.cast(x, tf.bool))(doc) # [bs, c_len]
        q_mask = Lambda(lambda x: tf.cast(x, tf.bool))(query)
        s_mask = Lambda(lambda x: tf.cast(x, tf.bool))(sent)
        # cont_len = Lambda(lambda x: tf.expand_dims(tf.reduce_sum(tf.cast(x, tf.int32), axis=1), axis=1))(c_mask)
        # ques_len = Lambda(lambda x: tf.expand_dims(tf.reduce_sum(tf.cast(x, tf.int32), axis=1), axis=1))(q_mask)
        x = context2query_attention(8 * self.config['hidden_size'], self.config['text1_maxlen'], self.config['text1_maxlen'], self.config['dropout_rate'])([s_rep, q_rep, s_mask, q_mask])
        # x = CrossATT(100, self.config['text1_maxlen'], self.config['text1_maxlen'], self.config['dropout_rate'])([s_rep, q_rep, s_mask, q_mask])
        show_layer_info('context2query_attention', x)
        
        l_att1 = AttLayer(2 * self.config['hidden_size'])(x)
        l_att2 = multiply([x, l_att1])
        l_att = Lambda( lambda x: K.sum(x, axis=1))(l_att2)
        show_layer_info('att 1', l_att)
        sentEncoder = Model([sent, query], l_att)

        query4 = TileLayer(self.sent_num)(query)
        query4_s = Reshape((self.sent_num, self.config['text1_maxlen']))(query4)
        show_layer_info('query4_s', query4_s)
        doc4 = Reshape((self.sent_num, self.config['text1_maxlen']))(doc)
        show_layer_info('doc4', doc4)

        concat = concatenate([query4_s, doc4])
        show_layer_info('concat 1', concat)
        out_model = TimeDistributed(Lambda(lambda x: sentEncoder([x[:,:self.config['text1_maxlen']], x[:, self.config['text1_maxlen']:]])), name="TimeDistributedhahaha")(concat)
        show_layer_info('out_model', out_model)


        l_att_sent1 = AttLayer(2 * self.config['hidden_size'])(out_model)
        l_att_sent2 = multiply([out_model, l_att_sent1])
        l_att_sent = Lambda( lambda x: K.sum(x, axis=1))(l_att_sent2)
        show_layer_info('att 2', l_att_sent)
        s_att_d = Dense(2 * self.config['hidden_size'], activation='relu')(l_att_sent)

        q_att1 = AttLayer(2 * self.config['hidden_size'])(q_rep)
        q_att2 = multiply([q_rep, q_att1])
        q_att = Lambda( lambda x: K.sum(x, axis=1))(q_att2)
        show_layer_info('att q', q_att)
        cross = multiply([q_att, s_att_d])
        # -1 flatten 
        cross_reshape = Reshape((-1, ))(cross)
        show_layer_info('Reshape', cross_reshape)

        # mm_k = Lambda(lambda x: K.tf.nn.top_k(x, k=self.config['topk'], sorted=True)[0])(cross_reshape)
        # show_layer_info('Lambda-topk', mm_k)

        pool1_flat_drop = Dropout(rate=self.config['dropout_rate'])(cross_reshape)
        show_layer_info('Dropout', pool1_flat_drop)

        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(pool1_flat_drop)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            #temp1 = Dense(self.config['hidden_size']//2, activation='relu')(pool1_flat_drop)
            #temp2 = Dense(self.config['hidden_size']//10, activation='relu')(temp1)
            out_ = Dense(1)(pool1_flat_drop)
        show_layer_info('Dense', out_)

        #model = Model(inputs=[query, doc, dpool_index], outputs=out_)
        model = Model(inputs=[query, doc], outputs=out_)
        return model
