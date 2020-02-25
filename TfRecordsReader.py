#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf

class TfRecordsReader:
    def __init__(
            self,
            tf_records_files=None,
            parse_function=None,
            batch_size=None,
            shuffle=True
    ):
        self.parse_function = parse_function
        self.batch_size = batch_size

        self.tf_records_files = tf_records_files
        dataset = tf.data.TFRecordDataset(self.tf_records_files)
        dataset = dataset.map(self.parse_function)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(self.batch_size)
        self.handle = tf.placeholder(tf.string, shape=[])
        t_iterator = tf.data.Iterator.from_string_handle(self.handle, dataset.output_types, dataset.output_shapes)
        self.next_element = t_iterator.get_next()
        self.iterator = dataset.make_initializable_iterator()