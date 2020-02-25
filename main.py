#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib

matplotlib.use('Agg')
import os
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from scipy import signal
import soundfile as sf
import time
import json
import matplotlib.pyplot as plt
from collections import namedtuple
import sys
import mir_eval
from TfRecordsReader import *
from TFRecordsParser import *
from LSTMModel import *
from BinaryMaskModel import *
from ChimeraNetwork import *
from BLSTMModel import *
from FileObserver import *

class ModelHandler():
    def __init__(self,
                 batch_size,
                 display_step,
                 num_input,
                 timesteps,
                 num_hidden,
                 layers,
                 d_vector,
                 alpha,
                 sources,
                 opt_params,
                 activation_function,
                 results_path,
                 graphs_path,
                 session_path,
                 tf_records_training_path,
                 tf_records_test_path,
                 max_speakers_train,
                 sample_rate,
                 stft_window,
                 stft_nperseg,
                 stft_noverlap,
                 globa_steps,
                 n_epochs,
                 phase_diff_threshold,
                 experiment_id,
                 csv_result_body,
                 n_speakers_train,
                 n_repeated_speakers_train,
                 files_per_speaker_test,
                 n_speakers_test_eval,
                 network,
                 load_session="false"):

        self.batch_size = batch_size
        self.display_step = display_step
        self.num_input = num_input
        self.timesteps = timesteps
        self.num_hidden = num_hidden
        self.layers = layers
        self.d_vector = d_vector
        self.alpha = alpha
        self.load_session = load_session
        self.sources = sources
        self.opt_params = opt_params
        self.activation_function = activation_function
        self.results_path = results_path
        self.graphs_path = graphs_path
        self.session_path = session_path
        self.tf_records_training_path = tf_records_training_path
        self.tf_records_test_path = tf_records_test_path
        self.max_speakers_train = max_speakers_train
        self.sample_rate = sample_rate
        self.stft_window = stft_window
        self.stft_nperseg = stft_nperseg
        self.stft_noverlap = stft_noverlap
        self.globa_steps = globa_steps
        self.n_epochs = n_epochs
        self.phase_diff_threshold = phase_diff_threshold
        self.experiment_id = experiment_id
        self.file_name = str(experiment_id)
        self.csv_result_body = csv_result_body
        self.n_speakers_train = n_speakers_train
        self.n_speakers_test_eval = n_speakers_test_eval
        self.n_repeated_speakers_train = n_repeated_speakers_train
        self.files_per_speaker_test = files_per_speaker_test
        self.network = network

        if not os.path.exists(self.graphs_path):
            os.makedirs(self.graphs_path)
        if not os.path.exists(self.session_path):
            os.makedirs(self.session_path)

    def create_body_string(self, step, train_acc, train_loss, test_acc, test_loss, test_SIR, test_SDR, test_SAR):

        self.csv_result_body = self.csv_result_body.replace("{steps}", str(step))
        self.csv_result_body = self.csv_result_body.replace("{train_acc}", str(train_acc))
        self.csv_result_body = self.csv_result_body.replace("{train_loss}", str(train_loss))
        self.csv_result_body = self.csv_result_body.replace("{test_acc}", str(test_acc))
        self.csv_result_body = self.csv_result_body.replace("{test_loss}", str(test_loss))
        self.csv_result_body = self.csv_result_body.replace("{test_SIR}", str(test_SIR))
        self.csv_result_body = self.csv_result_body.replace("{test_SDR}", str(test_SDR))
        self.csv_result_body = self.csv_result_body.replace("{test_SAR}", str(test_SAR))
        return self.csv_result_body

    def make_eval(self,model,sess,tf_records_reader_test,step,train_acc_mean,train_loss_mean,testing_handle):
        test_acc_mean = []
        test_loss_mean = []

        n_batches = self.n_speakers_test_eval * self.files_per_speaker_test / self.batch_size

        SDR_result = np.zeros((self.n_speakers_test_eval / self.files_per_speaker_test * n_batches))
        SIR_result = np.zeros((self.n_speakers_test_eval / self.files_per_speaker_test * n_batches))
        SAR_result = np.zeros((self.n_speakers_test_eval / self.files_per_speaker_test * n_batches))

        counter_audio = 0

        sess.run(tf_records_reader_test.iterator.initializer)

        string_file = ""
        result_data_details = "id,sdr,sir,sar\n"
        ##############################
        ######## EVALUATION ##########
        ##############################
        # Calculating SDR, SIR, SAR and creating raw audios.
        # Iterating over n_batches, depending on the parameters self.n_speakers_test_eval , self.files_per_speaker_test and self.batch_size
        for batch_ in range(n_batches):
            n_db_mag_X_0, n_db_mag_source, n_db_mag_interf, complex_X_0, MASK, VAD = sess.run(
                tf_records_reader_test.next_element, feed_dict={tf_records_reader_test.handle: testing_handle})

            l, acc, bmp, X_complex_output_, res_hat_, res_masked_ = sess.run(
                [model.loss, model.accuracy, model.MASK_hat, model.X_complex_output, model.res_hat, model.res_masked],
                feed_dict={model.X: np.concatenate([n_db_mag_source, n_db_mag_interf], axis=2), model.Y_true: MASK,
                           model.VAD: VAD, model.X_real: complex_X_0[:, :, :, 0],
                           model.X_imag: complex_X_0[:, :, :, 1], model.n_db_mag_X_0: n_db_mag_X_0})

            test_acc_mean.append(acc)
            test_loss_mean.append(l)

            for speaker in range(0,self.n_speakers_test_eval / self.files_per_speaker_test): #cantidad de locutores que existen en cada batch
                i = speaker * self.files_per_speaker_test
                #creating frames of size (files_per_speaker_test)
                a_hat = res_hat_[i:i + self.files_per_speaker_test, :, :, 0]  # self.timesteps , self.num_input , self.sources
                b_hat = res_hat_[i:i + self.files_per_speaker_test, :, :, 1]
                a_masked = res_masked_[i:i + self.files_per_speaker_test, :, :, 0]
                b_masked = res_masked_[i:i + self.files_per_speaker_test, :, :, 1]
                x = X_complex_output_[i:i + self.files_per_speaker_test, :, :]

                # 0 mix, 1 a, 2 b, 3 a_hat, 4 b_hat
                np_audio_signals = np.zeros([5, self.stft_window * self.files_per_speaker_test])

                #Creating raw signals
                for j in range(0,self.files_per_speaker_test):
                    start = j * self.stft_window
                    end = start + self.stft_window
                    _, np_audio_signals[0, start:end] = signal.istft(x[j], fs=self.sample_rate,
                        nperseg=self.stft_nperseg,
                        noverlap=self.stft_noverlap)
                    _, np_audio_signals[1, start:end] = signal.istft(a_masked[j], fs=self.sample_rate,
                        nperseg=self.stft_nperseg,
                        noverlap=self.stft_noverlap)
                    _, np_audio_signals[2, start:end] = signal.istft(b_masked[j], fs=self.sample_rate,
                        nperseg=self.stft_nperseg,
                        noverlap=self.stft_noverlap)
                    _, np_audio_signals[3, start:end] = signal.istft(a_hat[j], fs=self.sample_rate,
                        nperseg=self.stft_nperseg,
                        noverlap=self.stft_noverlap)
                    _, np_audio_signals[4, start:end] = signal.istft(b_hat[j], fs=self.sample_rate,
                        nperseg=self.stft_nperseg,
                        noverlap=self.stft_noverlap)
                #Calculating SDR,SIR,SAR
                (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(
                    np_audio_signals[1:3, :] + np.finfo(np.float).eps, np_audio_signals[3:5, :] + np.finfo(np.float).eps)
                
                string_file += "{} {} {} {} \n".format(sdr, sir, sar, perm)

                SDR_result[counter_audio] = sdr[perm.tolist().index(0)]
                SIR_result[counter_audio] = sir[perm.tolist().index(0)]
                SAR_result[counter_audio] = sar[perm.tolist().index(0)]

                #Saving files
                sf.write(self.results_path + self.file_name + "/rec_audios/{}_x_mix.wav".format(counter_audio),
                         np_audio_signals[0, :], self.sample_rate)
                sf.write(self.results_path + self.file_name + "/rec_audios/{}_x_a.wav".format(counter_audio),
                         np_audio_signals[1, :], self.sample_rate)
                sf.write(self.results_path + self.file_name + "/rec_audios/{}_x_b.wav".format(counter_audio),
                         np_audio_signals[2, :], self.sample_rate)
                sf.write(self.results_path + self.file_name + "/rec_audios/{}_x_a_hat.wav".format(counter_audio),
                         np_audio_signals[3, :], self.sample_rate)
                sf.write(self.results_path + self.file_name + "/rec_audios/{}_x_b_hat.wav".format(counter_audio),
                         np_audio_signals[4, :], self.sample_rate)

                result_data_details += "{},{},{},{}\n".format(
                    counter_audio,
                    sdr[perm.tolist().index(0)],
                    sir[perm.tolist().index(0)],
                    sar[perm.tolist().index(0)]
                )

                print("SDR_mean():", SDR_result.mean())
                print("SIR_mean():", SIR_result.mean())
                print("SAR_mean():", SAR_result.mean())
                print(SDR_result)
                counter_audio += 1

        file = open(self.results_path + self.file_name + '/audio_details.csv', "w")
        file.write(result_data_details)
        file.close()

        body_string = self.create_body_string(
            step,
            np.array(train_acc_mean).mean(),
            np.array(train_loss_mean).mean(),
            np.array(test_acc_mean).mean(),
            np.array(test_loss_mean).mean(),
            SIR_result.mean(),
            SDR_result.mean(),
            SAR_result.mean()
        )

        file = open(self.results_path + self.file_name + '/condensed_result.csv', "w")
        file.write(body_string)
        file.close()


    def run_model(self):

        path = self.tf_records_training_path
        test_path = self.tf_records_test_path

        # number of tfrecords files, default is 8
        files = [0, 1, 2, 3, 4, 5, 6, 7]
        tf_records_files = [path + '{}.tfrecords'.format(global_step) for global_step in files]
        test_tf_records_files = [test_path + '0.tfrecords']


        if(self.network == "BLSTMModel"):
            ###BinaryMaskModel
            print("Using BLSTMModel")
            model = BLSTMModel(
                num_input=self.num_input,
                timesteps=self.timesteps,
                num_hidden=self.num_hidden,
                layers=self.layers,
                sources=self.sources,
                optimizer = self.opt_params.optimizer,
                learning_rate = self.opt_params.learning_rate,
                batch_size = self.batch_size,
                momentum = self.opt_params.momentum,
                forget_bias= 0.0 if self.load_session == "true" else 1.0)  # 0.0 if self.load_session == "true" else 1.0
        elif(self.network == "ChimeraNetwork"):
            ###BinaryMaskModel
            print("Using ChimeraNetwork")
            model = ChimeraNetwork(
                num_input=self.num_input,
                timesteps=self.timesteps,
                num_hidden=self.num_hidden,
                layers=self.layers,
                d_vector=self.d_vector,
                sources=self.sources,
                activation_function=self.activation_function,
                optimizer = self.opt_params.optimizer,
                learning_rate = self.opt_params.learning_rate,
                batch_size = self.batch_size,
                alpha = self.alpha,
                momentum = self.opt_params.momentum,
                forget_bias= 0.0 if self.load_session == "true" else 1.0)  # 0.0 if self.load_session == "true" else 1.0


        file_observer = FileObserver(tf_records_files)

        tfRecordsParser = TFRecordsParser(self.num_input, self.timesteps, self.sources)

        tf_records_reader_training = TfRecordsReader(
            tf_records_files=tf_records_files,
            parse_function=tfRecordsParser.parse_function,
            batch_size=self.batch_size
        )
        tf_records_reader_test = TfRecordsReader(
            tf_records_files=test_tf_records_files,
            parse_function=tfRecordsParser.parse_function,
            batch_size=self.batch_size,
            shuffle = False
        )

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        tf_records_size_training = self.n_speakers_train * self.n_repeated_speakers_train
        steps_to_complete_tf_record = tf_records_size_training / float(self.batch_size)

        ##############Config paths
        speakers_to_save = 2

        total_samples = speakers_to_save * self.files_per_speaker_test

        if not os.path.exists(self.results_path + self.file_name + '/'):
            os.makedirs(self.results_path + self.file_name + '/')
        if not os.path.exists(self.results_path + self.file_name + '/plot/'):
            os.makedirs(self.results_path + self.file_name + '/plot/')
        if not os.path.exists(self.results_path + self.file_name + '/rec_audios/'):
            os.makedirs(self.results_path + self.file_name + '/rec_audios/')


        for speaker in range(speakers_to_save):
            for global_step in range(self.files_per_speaker_test):
                if not os.path.exists(self.results_path + self.file_name + '/plot/{}/{}/'.format(speaker, global_step)):
                    os.makedirs(self.results_path + self.file_name + '/plot/{}/{}/'.format(speaker, global_step))

        for global_step in range(0, total_samples, self.files_per_speaker_test):
            if not os.path.exists(self.results_path + self.file_name + '/rec_audios/{}/'.format(global_step)):
                os.makedirs(self.results_path + self.file_name + '/rec_audios/{}/'.format(global_step))
        ##############

        with tf.Session() as sess:
            print("V15")
            train_writer = tf.summary.FileWriter(self.graphs_path + '{}_train'.format(self.file_name))
            test_writer = tf.summary.FileWriter(self.graphs_path + '{}_test'.format(self.file_name))

            train_writer.add_graph(sess.graph)
            sess.run(init_op)
            training_handle = sess.run(tf_records_reader_training.iterator.string_handle())
            testing_handle = sess.run(tf_records_reader_test.iterator.string_handle())

            saver = tf.train.Saver()
            step = 0

            if self.load_session == "true":
                file = open(self.session_path + '{}.stp'.format(self.file_name), "r")
                step = int(file.read().strip())
                print("STEP:", step)
                file.close()
                #Restoring session
                saver.restore(sess, self.session_path + '{}/model_weights.ckpt'.format(self.file_name))
                ######### 2000 = tfrecord file size, 8 = tfrecord files
                total_steps = (self.n_speakers_train * self.n_repeated_speakers_train) * 8 * self.n_epochs * self.globa_steps / self.batch_size
                ####remaining global steps                                 2000 = tfrecord file size, 8 = tfrecord files
                remaining = int(((total_steps - step) * self.batch_size) / ((self.n_speakers_train * self.n_repeated_speakers_train) * 8 * self.n_epochs)) 

                print("Remaining global_steps:",remaining)
                new_step = (self.n_speakers_train * self.n_repeated_speakers_train) * 8 * self.n_epochs * (self.globa_steps - remaining) / self.batch_size
                print("NEW STEP:", new_step)
                step = new_step
                self.globa_steps = remaining

            train_acc_mean = []
            train_loss_mean = []

            for global_step in range(0, self.globa_steps):
                ####################################
                ###########  TRAINING ##############
                ####################################
                for epoch in range(0, self.n_epochs):
                    file_observer.wait_if_not_exist_files()
                    sess.run(tf_records_reader_training.iterator.initializer)
                    last_tf_records_file_loaded = 0
                    train_acc_mean = []
                    train_loss_mean = []
                    while True:
                        try:
                            n_db_mag_X_0, n_db_mag_source, n_db_mag_interf, complex_X_0, MASK, VAD = sess.run(
                                tf_records_reader_training.next_element,
                                feed_dict={tf_records_reader_training.handle: training_handle})

                            _, l, acc, lg, bmp, train_summary = sess.run(
                                [model.optimizer, model.loss, model.accuracy, model.y_pred, model.MASK_hat, model.summary],
                                feed_dict={model.X: np.concatenate([n_db_mag_source, n_db_mag_interf], axis=2),
                                           model.Y_true: MASK, model.VAD: VAD, model.X_real: complex_X_0[:, :, :, 0],
                                           model.X_imag: complex_X_0[:, :, :, 1], model.n_db_mag_X_0: n_db_mag_X_0,
                                           model.n_db_mag_X_0: n_db_mag_X_0})

                            train_acc_mean.append(acc)
                            train_loss_mean.append(l)

                            train_writer.add_summary(train_summary, step)
                            if step % self.display_step == 0 or step == 1:
                                print(lg[0][0])
                                print(bmp.shape)
                                print(np.sum(np.count_nonzero(bmp, axis=1), axis=0))
                                print('Step {}, globa_steps {} Minibatch Loss:{} Acc:{}'.format(str(step), str(global_step), str(l),
                                                                                          str(acc)))
                            step += 1
                            if step % steps_to_complete_tf_record == 0:
                                if epoch == self.n_epochs - 1:
                                    os.remove(path + "{}.tfrecords".format(last_tf_records_file_loaded))
                                last_tf_records_file_loaded += 1

                        except tf.errors.OutOfRangeError:
                            break;

                    print('epoch acc mean :{}'.format(str(np.array(train_acc_mean).mean())))
                    file = open(self.session_path + '{}.stp'.format(self.file_name), "w")
                    file.write(str(step))
                    file.close()
                    save_path = saver.save(sess, self.session_path + '{}/model_weights.ckpt'.format(self.file_name))

                ####################################
                ###########  TESTING  ##############
                ####################################
                #if global_step % 1 == 0:
                sess.run(tf_records_reader_test.iterator.initializer)
                while True:
                    try:
                        n_db_mag_X_0, n_db_mag_source, n_db_mag_interf, complex_X_0, MASK, VAD = sess.run(
                            tf_records_reader_test.next_element,
                            feed_dict={tf_records_reader_test.handle: testing_handle})
                        if(type(model) is ChimeraNetwork):
                            res_hat_, l, acc, test_summary, Z_, Y_ = sess.run(
                                [model.res_hat, model.loss, model.accuracy, model.summary, model.Z_res, model.Y_res],
                                feed_dict={model.X: np.concatenate([n_db_mag_source, n_db_mag_interf], axis=2),
                                           model.Y_true: MASK, model.VAD: VAD, model.X_real: complex_X_0[:, :, :, 0],
                                           model.X_imag: complex_X_0[:, :, :, 1], model.n_db_mag_X_0: n_db_mag_X_0})
                        else:
                            res_hat_, l, acc, test_summary= sess.run(
                                [model.res_hat, model.loss, model.accuracy, model.summary],
                                feed_dict={model.X: np.concatenate([n_db_mag_source, n_db_mag_interf], axis=2),
                                           model.Y_true: MASK, model.VAD: VAD, model.X_real: complex_X_0[:, :, :, 0],
                                           model.X_imag: complex_X_0[:, :, :, 1], model.n_db_mag_X_0: n_db_mag_X_0})

                        print('TEST Step {}, globa_steps {} Minibatch Loss:{} Acc:{}'.format(str(step), str(global_step), str(l),
                                                                                       str(acc)))
                        test_writer.add_summary(test_summary, step)
                    except tf.errors.OutOfRangeError:
                        break;

                    #Creating audios of two speakers just to check
                    for sample in range(0, speakers_to_save * self.files_per_speaker_test, self.files_per_speaker_test):
                        a_hat = res_hat_[sample: sample + self.files_per_speaker_test, :, :,0]  # self.timesteps , self.num_input , self.sources
                        b_hat = res_hat_[sample: sample + self.files_per_speaker_test, :, :,1]
                        x_a_hat_list = []
                        x_b_hat_list = []

                        for j in range(0, self.files_per_speaker_test):
                            _, x_a_hat = signal.istft(a_hat[j], fs=self.sample_rate, nperseg=self.stft_nperseg,
                                                      noverlap=self.stft_noverlap)
                            _, x_b_hat = signal.istft(b_hat[j], fs=self.sample_rate, nperseg=self.stft_nperseg,
                                                      noverlap=self.stft_noverlap)
                            x_a_hat_list += x_a_hat.tolist()
                            x_b_hat_list += x_b_hat.tolist()
                        sf.write(
                            self.results_path + self.file_name + '/rec_audios/{}/{}_x_a_hat.wav'.format(sample, global_step),
                            x_a_hat_list, self.sample_rate)
                        sf.write(
                            self.results_path + self.file_name + '/rec_audios/{}/{}_x_b_hat.wav'.format(sample, global_step),
                            x_b_hat_list, self.sample_rate)

            
            self.make_eval(model,sess,tf_records_reader_test,step,train_acc_mean,train_loss_mean,testing_handle)

def main():
    # silences Tensorflow boot logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Using just one GPU in case of GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]  # '1'

    # configuration file path
    configuration_file = str(sys.argv[1])
    if configuration_file == "":
        print("ERROR: you need to define param: configuration <configuration_file>.json ")
        exit(0)
   

    PARAMS = None
    # load and convert configuration file to an object
    with open(configuration_file, 'r') as f:
        f = f.read()
        PARAMS = json.loads(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

    #creamos el folder pricipal
    #Creating main path
    if not os.path.exists(PARAMS.PATHS.main_path):
        os.makedirs(PARAMS.PATHS.main_path)

    if not os.path.exists(PARAMS.PATHS.main_path+"experiment_id.npy"):
        np.save(PARAMS.PATHS.main_path+"experiment_id.npy",np.array([0]))

    #aqui preguntar si debemos cargar algo que existe o vamos a hacer uno nuevo
    # Load experiment_id is necessary
    if len(sys.argv) == 5 and sys.argv[3] == "true":
        experiment_id = int(sys.argv[4])
    else:
        experiment_id = np.load(PARAMS.PATHS.main_path+"experiment_id.npy")[0]
        experiment_id +=1
        np.save(PARAMS.PATHS.main_path+"experiment_id.npy",np.array([experiment_id]))
        
    print("experiment_id",experiment_id)
    print(vars(PARAMS))

    NPERSEG = PARAMS.DATA_GENERATOR.stft_nperseg
    NOVERLAP = PARAMS.DATA_GENERATOR.stft_noverlap
    WINDOW = PARAMS.DATA_GENERATOR.stft_window
    # calculating window size
    TIME_STEP = int(np.ceil((WINDOW - NPERSEG) / (NPERSEG - NOVERLAP))) + 3
    INPUT = int((NPERSEG / 2.0) + 1)

    print("input", INPUT)
    print("timesteps", TIME_STEP)

    ##creating header and body for the result
    csv_result_header = []
    csv_result_data = []

    for attr, value in vars(PARAMS.DATA_GENERATOR).items():
        csv_result_header.append(attr)
        csv_result_data.append(value)

    for attr, value in vars(PARAMS.TRAINING).items():
        if attr != "opt_params":
            csv_result_header.append(attr)
            csv_result_data.append(value)
        else:
            for attr_2, value_2 in vars(value).items():
                csv_result_header.append(attr_2)
                csv_result_data.append(value_2)

    if not os.path.exists(PARAMS.PATHS.main_path):
        os.makedirs(PARAMS.PATHS.main_path)


    #creating name of the result folder
    file_content = ""

    if not os.path.exists(PARAMS.PATHS.main_path+"table_id_experiment.csv"):
        file_content +=','.join(csv_result_header)+",experiment_id" + "\n"
    
    file_content +=','.join([ str(i) for i in csv_result_data])+",{}".format(experiment_id)+"\n"
    
    file = open(PARAMS.PATHS.main_path+"table_id_experiment.csv","a")
    file.write(file_content)
    file.close()


    csv_result_body = ','.join(csv_result_header) + ",steps,train_acc,train_loss,test_acc,test_loss,test_SIR,test_SDR,test_SAR\n"
    csv_result_body += ','.join([ str(i) for i in csv_result_data]) + ",{steps},{train_acc},{train_loss},{test_acc},{test_loss},{test_SIR},{test_SDR},{test_SAR}"

    if (PARAMS.TRAINING.network != "BLSTMModel" and PARAMS.TRAINING.network != "ChimeraNetwork"):
        print("Error loading network, only BLSTMModel and ChimeraNetwork available.")
        exit(0)

    model = ModelHandler(
        batch_size=PARAMS.TRAINING.batch_size,
        display_step=PARAMS.TRAINING.display_step,
        num_input=INPUT,
        timesteps=TIME_STEP,
        num_hidden=PARAMS.TRAINING.num_hidden,
        layers=PARAMS.TRAINING.layers,
        d_vector=PARAMS.TRAINING.d_vector,
        alpha=PARAMS.TRAINING.alpha,
        load_session=sys.argv[3],
        sources=2,
        opt_params=PARAMS.TRAINING.opt_params,
        activation_function=PARAMS.TRAINING.activation_function,
        results_path=PARAMS.PATHS.main_path + "results/",
        graphs_path=PARAMS.PATHS.main_path + "graphs/",
        session_path=PARAMS.PATHS.main_path + "session/",
        tf_records_training_path=PARAMS.PATHS.tf_records_training_path,
        tf_records_test_path=PARAMS.PATHS.tf_records_test_path,
        max_speakers_train=PARAMS.DATA_GENERATOR.max_speakers_train,
        sample_rate=PARAMS.DATA_GENERATOR.sample_rate,
        stft_window=PARAMS.DATA_GENERATOR.stft_window,
        stft_nperseg=PARAMS.DATA_GENERATOR.stft_nperseg,
        stft_noverlap=PARAMS.DATA_GENERATOR.stft_noverlap,
        globa_steps=PARAMS.TRAINING.globa_steps,
        n_epochs=PARAMS.TRAINING.n_epochs,
        phase_diff_threshold=PARAMS.DATA_GENERATOR.phase_diff_threshold,
        experiment_id=experiment_id,
        csv_result_body=csv_result_body,
        n_speakers_train=PARAMS.DATA_GENERATOR.n_speakers_train,
        n_repeated_speakers_train=PARAMS.DATA_GENERATOR.n_repeated_speakers_train,
        files_per_speaker_test=PARAMS.DATA_GENERATOR.files_per_speaker_test,
        n_speakers_test_eval=PARAMS.DATA_GENERATOR.n_speakers_test_eval,
        network = PARAMS.TRAINING.network
    )
    if not os.path.exists(model.results_path + model.file_name + '/'):
        os.makedirs(model.results_path + model.file_name + '/')
        file_config = open(model.results_path + model.file_name + '/config.json', "w")
        file_config.write(f)
        file_config.close()
    
    model.run_model()

if __name__ == "__main__":
    main()