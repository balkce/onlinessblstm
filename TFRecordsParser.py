#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

class TFRecordsParser():
    def __init__(self, INPUT, TIME_STEP, SOURCES):
        self.TIME_STEP = TIME_STEP
        self.INPUT = INPUT
        self.SOURCES = SOURCES

    def parse_function(self, proto):
        features = tf.parse_single_example(proto,
            features={
            'n_db_mag_X_0': tf.FixedLenFeature([], tf.string),
            'n_db_mag_ref': tf.FixedLenFeature([], tf.string),
            'n_db_mag_interf': tf.FixedLenFeature([], tf.string),
            'complex_X_0': tf.FixedLenFeature([], tf.string),
            'MASK': tf.FixedLenFeature([], tf.string),
            'VAD': tf.FixedLenFeature([], tf.string)
            })

        n_db_mag_X_0 = tf.decode_raw(features['n_db_mag_X_0'], tf.float32)
        n_db_mag_X_0 = tf.reshape(n_db_mag_X_0, (self.INPUT, self.TIME_STEP))
        n_db_mag_X_0 = tf.transpose(n_db_mag_X_0)

        n_db_mag_source = tf.decode_raw(features['n_db_mag_ref'], tf.float32)
        n_db_mag_source = tf.reshape(n_db_mag_source, (self.INPUT, self.TIME_STEP))
        n_db_mag_source = tf.transpose(n_db_mag_source)

        n_db_mag_interf = tf.decode_raw(features['n_db_mag_interf'], tf.float32)
        n_db_mag_interf = tf.reshape(n_db_mag_interf, (self.INPUT, self.TIME_STEP))
        n_db_mag_interf = tf.transpose(n_db_mag_interf)

        complex_X_0 = tf.decode_raw(features['complex_X_0'], tf.float32)
        complex_X_0 = tf.reshape(complex_X_0, (self.INPUT, self.TIME_STEP, 2))
        complex_X_0 = tf.transpose(complex_X_0, (1, 0, 2))

        MASK = tf.decode_raw(features['MASK'], tf.uint8)
        MASK = tf.reshape(MASK, (self.INPUT, self.TIME_STEP, self.SOURCES))
        MASK = tf.transpose(MASK, (1, 0, 2))
        MASK = tf.cast(MASK, tf.float32)

        VAD = tf.decode_raw(features['VAD'], tf.uint8)
        VAD = tf.reshape(VAD, (self.INPUT, self.TIME_STEP))
        VAD = tf.transpose(VAD)
        VAD = tf.cast(VAD, tf.float32)

        return n_db_mag_X_0, n_db_mag_source, n_db_mag_interf, complex_X_0, MASK, VAD