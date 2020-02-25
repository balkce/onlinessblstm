#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from LSTMModel import *

class BinaryMaskModel(LSTMModel):
    def __init__(self,
                 num_input,
                 timesteps,
                 num_hidden,
                 layers,
                 optimizer,
                 learning_rate,
                 momentum,
                 batch_size,
                 sources):

        LSTMModel.__init__(self,num_input = num_input,
                 timesteps = timesteps,
                 num_hidden = num_hidden,
                 layers = layers,
                 optimizer = optimizer,
                 learning_rate = learning_rate,
                 momentum = momentum,
                 batch_size = batch_size)
        self.sources = sources

        #Outputs
        self.MASK_hat = None
        self.MASK_hat_a = None
        self.MASK_hat_b = None

        self.res_hat = None
        self.X_complex_output = None
        self.res_masked = None

    def def_output(self):
        with tf.name_scope('output'):
            ############
            self.MASK_hat = tf.cast(tf.equal(tf.reduce_max(self.y_pred, axis=2, keepdims=True), self.y_pred), tf.float32)
            MASK_hat_rs = tf.reshape(self.MASK_hat, (-1, self.timesteps, self.num_input, self.sources))
            MASK_hat_T = tf.transpose(MASK_hat_rs, (0, 2, 1, 3))
            self.MASK_hat_a, self.MASK_hat_b = tf.split(MASK_hat_T, 2, axis=3)

            ############Audio output
            X_complex = tf.reshape(self.X_complex, shape=[-1, self.timesteps * self.num_input, 1])
            X_complex_double = tf.concat([X_complex, X_complex], axis=2)  # -1, self.timesteps * self.num_input , 2

            res_hat = tf.multiply(X_complex_double, tf.cast(self.MASK_hat, tf.complex64))

            res_hat = tf.reshape(res_hat, (-1, self.timesteps, self.num_input, self.sources))

            self.res_hat = tf.transpose(res_hat, (0, 2, 1, 3))

            Y_true_rs = tf.reshape(self.Y_true, (-1, self.timesteps * self.num_input, self.sources))

            X_complex_output = tf.reshape(self.X_complex, shape=[-1, self.timesteps, self.num_input])
            self.X_complex_output = tf.transpose(X_complex_output, (0, 2, 1))

            res_masked = tf.multiply(X_complex_double, tf.cast(Y_true_rs, tf.complex64))

            res_masked = tf.reshape(res_masked, (-1, self.timesteps, self.num_input, self.sources))

            self.res_masked = tf.transpose(res_masked, (0, 2, 1, 3))