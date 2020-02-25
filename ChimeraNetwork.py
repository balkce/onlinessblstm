#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from BinaryMaskModel import * 
from tensorflow.contrib import rnn

class ChimeraNetwork(BinaryMaskModel):
    def __init__(self,
                 num_input,
                 timesteps,
                 num_hidden,
                 layers,
                 d_vector,
                 sources,
                 activation_function,
                 optimizer,
                 learning_rate,
                 batch_size,
                 alpha,
                 momentum,
                 forget_bias):
        BinaryMaskModel.__init__(self,
                                 num_input=num_input,
                                 timesteps=timesteps,
                                 num_hidden=num_hidden,
                                 layers=layers,
                                 optimizer=optimizer,
                                 learning_rate=learning_rate,
                                 momentum=momentum,
                                 batch_size=batch_size,
                                 sources=sources)

        dict_af = {"relu": tf.nn.relu, "tanh": tf.nn.tanh, "sigmoid": tf.nn.sigmoid}

        self.d_vector = d_vector
        self.activation_function = dict_af[activation_function]
        self.forget_bias = forget_bias
        self.alpha = alpha

        # Model
        self.Z = None
        self.y_pred = None
        self.embeding_dc_rs_normalized = None

        self.build()

    def __lstm_cell(self, num_hidden):
        return tf.contrib.rnn.LSTMCell(
            num_hidden, forget_bias=self.forget_bias,
            initializer=tf.contrib.layers.xavier_initializer(),
            activation=tf.tanh)

    def def_params(self):
        initializer = tf.contrib.layers.xavier_initializer()
        self.weights = {
            'out': tf.get_variable('weights1', [2 * self.num_hidden, self.num_input * self.d_vector],
                                   initializer=initializer),
            'out2': tf.get_variable('weights2', [self.d_vector, self.sources], initializer=initializer)
        }
        self.biases = {
            'out': tf.get_variable('bias1', [self.num_input * self.d_vector], initializer=initializer),
            'out2': tf.get_variable('bias2', [self.sources], initializer=initializer)
        }

    def def_model(self):
        self.Z = self.__common_model(self.X, self.timesteps, self.num_hidden, self.layers, self.weights['out'],
                                     self.biases['out'])
        self.y_pred = self.__inference_head_model(self.Z, self.d_vector, self.weights['out2'], self.biases['out2'])
        self.embeding_dc_rs_normalized = self.__clustering_head_model(self.Z, self.d_vector)

    def __common_model(self, X, timesteps, num_hidden, layers, w1, b1):
        x = tf.unstack(X, timesteps, 1)
        fw_lstm_cells_encoder = [self.__lstm_cell(num_hidden) for i in range(layers)]
        bw_lstm_cells_encoder = [self.__lstm_cell(num_hidden) for i in range(layers)]
        outputs, output_state_fw, output_state_bw = rnn.stack_bidirectional_rnn(fw_lstm_cells_encoder,
                                                                                bw_lstm_cells_encoder, x,
                                                                                dtype=tf.float32)
        # check size
        print("outputs len:", len(outputs))
        print("outputs[0].shape:", outputs[0].shape)
        ################
        outputs = tf.reshape(outputs, [timesteps, -1, num_hidden * 2])
        print("R_outputs[0].shape:", outputs.shape)
        # Sort, first batch dimension
        sorted_outputs = tf.transpose(outputs, (1, 0, 2))
        print("sorted_outputs.shape:", sorted_outputs)

        # list is reshaped in order to multiply with the matrix
        ######################################batch * timesteps, num_hidden * 2
        outputs = tf.reshape(sorted_outputs, [-1, num_hidden * 2])

        # Vector Z is calculated
        return tf.matmul(outputs, w1) + b1  # batch * timesteps, self.num_input  * self.d_vector

    def __inference_head_model(self, Z, d_vector, w2, b2):
        # Mask inference head
        
        ##################batch * self.timesteps * self.num_input, self.d_vector
        V = tf.reshape(Z, [-1, d_vector])
        
        y_pred = tf.matmul(V, w2) + b2  # batch * timesteps * self.num_input, self.sources
        #
        # Probabilties are calculated with softmax, in axis 1
        y_pred = tf.nn.softmax(y_pred, axis=1)
        ####################################### self.batch_size, self.timesteps * self.num_input , self.sources
        return tf.reshape(y_pred, shape=[-1, self.timesteps * self.num_input, self.sources])

    def __clustering_head_model(self, Z, d_vector):
        # Deep clustering head
        embeding_dc = self.activation_function(Z)
        ######################################### batch_size * self.timesteps * self.num_input, self.d_vector
        embeding_dc_rs = tf.reshape(embeding_dc, [-1, d_vector])
        ############Norm in D vector
        return tf.nn.l2_normalize(embeding_dc_rs, 1)

    def def_loss(self):
        """ Defines loss function """

        ##Applying VAD
        ##################################### batch_size * self.timesteps * self.num_input , self.sources
        Y_true_rs = tf.reshape(self.Y_true, shape=[-1, self.sources])
        Y_true_rs_vad = tf.transpose(tf.multiply(tf.transpose(Y_true_rs), self.VAD_rs))
        ###############################self.batch_size, self.timesteps * self.num_input , self.sources
        self.Y_true_vad = tf.reshape(Y_true_rs_vad, shape=[-1, self.timesteps * self.num_input, self.sources])

        ##########################################batch_size * self.timesteps * self.num_input , self.sources
        y_pred_rs = tf.reshape(self.y_pred, shape=[-1, self.sources])
        y_pred_rs_vad = tf.transpose(tf.multiply(tf.transpose(y_pred_rs), self.VAD_rs))
        ################################## self.batch_size, self.timesteps * self.num_input , self.sources
        self.y_pred_vad = tf.reshape(y_pred_rs_vad, shape=[-1, self.timesteps * self.num_input, self.sources])

        #########Loss 1
        ##############Apply VAD to embedding
        embedding_dc_rs_normalized_vad = tf.transpose(tf.multiply(tf.transpose(self.embeding_dc_rs_normalized),
                                                                  self.VAD_rs))  # batch_size * self.timesteps * self.num_input, self.d_vector
        self.embeddings_v = tf.reshape(embedding_dc_rs_normalized_vad,
                                       shape=[-1, self.timesteps * self.num_input, self.d_vector])

        t1 = tf.nn.l2_loss(tf.matmul(tf.transpose(self.embeddings_v, (0, 2, 1)), self.embeddings_v))
        t2 = tf.nn.l2_loss(tf.matmul(tf.transpose(self.embeddings_v, (0, 2, 1)), self.Y_true_vad))
        t3 = tf.nn.l2_loss(tf.matmul(tf.transpose(self.Y_true_vad, (0, 2, 1)), self.Y_true_vad))

        loss_1 = (t1 + t3 - 2 * t2) / self.batch_size

        #########Loss 2
        ##############Apply VAD to signal of microphone 0
        n_db_mag_X_0_rs = tf.reshape(self.n_db_mag_X_0, [-1, 1])
        n_db_mag_X_0_rs_vad = tf.transpose(tf.multiply(tf.transpose(n_db_mag_X_0_rs), self.VAD_rs))
        n_db_mag_X_0_vad = tf.reshape(n_db_mag_X_0_rs_vad, shape=[-1, self.timesteps, self.num_input])

        y1, y2 = tf.split(self.y_pred_vad, 2, axis=2)
        y1 = tf.reshape(y1, shape=[-1, self.timesteps, self.num_input])
        y2 = tf.reshape(y2, shape=[-1, self.timesteps, self.num_input])

        Y_v1, Y_v2 = tf.split(self.Y_true_vad, 2, axis=2)
        Y_v1 = tf.reshape(Y_v1, shape=[-1, self.timesteps, self.num_input])
        Y_v2 = tf.reshape(Y_v2, shape=[-1, self.timesteps, self.num_input])

        # https://github.com/pchao6/LSTM_PIT_Speech_Separation/blob/master/blstm.py
        loss_2 = tf.reduce_mean(tf.reduce_mean(
            tf.reduce_sum(tf.pow(tf.multiply(Y_v1 - y1, n_db_mag_X_0_vad), 2), 1) +
            tf.reduce_sum(tf.pow(tf.multiply(Y_v2 - y2, n_db_mag_X_0_vad), 2), 1), 1))

        self.loss = self.alpha * loss_1 + (1 - self.alpha) * loss_2

        ###################### Just for plot 3 principal components:
        ########################################### batch_size, self.timesteps * self.num_input, self.d_vector
        self.Z_res = tf.reshape(self.embeddings_v, shape=[-1, self.timesteps * self.num_input, self.d_vector])
        ################################### self.batch_size, self.timesteps * self.num_input, self.sources
        self.Y_res = tf.reshape(self.Y_true_vad, shape=[-1, self.timesteps * self.num_input, self.sources])
        ######################

    def add_summaries(self):
        """ Adds summaries for Tensorboard """
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

            tf.summary.histogram('W0', self.weights['out'])
            tf.summary.histogram('B0', self.biases['out'])

            tf.summary.histogram('W1', self.weights['out2'])
            tf.summary.histogram('B1', self.biases['out2'])
            self.summary = tf.summary.merge_all()