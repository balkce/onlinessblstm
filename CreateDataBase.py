#!/usr/bin/env python
# -*- coding: utf-8 -*-

import soundfile as sf
from scipy import signal
import tensorflow as tf
import os
import sys
from tqdm import tqdm
import random
import math
import FileObserver as fo
import BeamFormer2 as BF
from AudioUtils import *
import json
from collections import namedtuple
import mir_eval
import csv
class TFRecordsConverterDC:
    def __init__(self):
        self.features_list = []

    def set_features(self, features):
        self.features_list = features
        return self

    def convert_to_tf(self, filename):
        if len(self.features_list) > 0:
            self._convert_to(filename)

    def int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _convert_to(self, filename):

        file_name = filename.split('/')[-1]

        path = '/'.join(filename.split('/')[0:-1]) + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        path_tmp = '/'.join(filename.split('/')[0:-1]) + "/tmp/"
        if not os.path.exists(path_tmp):
            os.makedirs(path_tmp)

        print('Writing', path_tmp + file_name)
        writer = tf.python_io.TFRecordWriter(path_tmp + file_name)
        for f in self.features_list:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature=f
                )
            )
            writer.write(example.SerializeToString())
        writer.close()
        os.rename(path_tmp + file_name, path + file_name)


class GenerateDatabase:
    def __init__(self,
                 data_base,
                 training_tf_records_files,
                 n_speakers_in_training_file,
                 n_repeated_speaker_in_training_file,
                 max_mixed_speakers_in_training_file,
                 min_mixed_speakers_in_training_file,
                 tf_records_test_path,
                 n_speakers_in_test_file,
                 n_files_per_speaker_in_training_file,
                 max_mixed_speakers_in_test_file,
                 min_mixed_speakers_in_test_file,
                 phase_diff_threshold,
                 training_buffer_size,
                 nframes,
                 nperseg,
                 noverlap,
                 window_size,
                 audio_folder_path,
                 sample_rate,
                 microphones_train,
                 microphones_test
                 ):

        self.training_tf_records_files = training_tf_records_files
        self.n_speakers_in_training_file = n_speakers_in_training_file
        self.n_repeated_speaker_in_training_file = n_repeated_speaker_in_training_file
        self.max_mixed_speakers_in_training_file = max_mixed_speakers_in_training_file
        self.min_mixed_speakers_in_training_file = min_mixed_speakers_in_training_file

        self.tf_records_test_path = tf_records_test_path
        self.n_speakers_in_test_file = n_speakers_in_test_file
        self.n_files_per_speaker_in_training_file = n_files_per_speaker_in_training_file
        self.max_mixed_speakers_in_test_file = max_mixed_speakers_in_test_file
        self.min_mixed_speakers_in_test_file = min_mixed_speakers_in_test_file

        self.phase_diff_threshold = float(phase_diff_threshold) * math.pi / 180.0
        self.training_buffer_size_limit = training_buffer_size
        self.nframes = nframes

        self.nperseg = nperseg
        self.noverlap = noverlap
        self.window_size = window_size

        self.time_step = int(np.ceil((self.window_size - self.nperseg) / (self.nperseg - self.noverlap))) + 3
        self.input_size = int((self.nperseg / 2.0) + 1)

        self.window_length = self.nframes * ((self.window_size / self.nframes) + 4 + 3)
        self.window_length_test = self.nframes * (
                ((self.window_size * self.n_files_per_speaker_in_training_file) / self.nframes) + 4 + 3)

        self.audio_folder_path = audio_folder_path  # "/home/ar/source_separation/LibriSpeech_6000/"
        self.microphones_train = microphones_train
        self.microphones_test = microphones_test 
        self.ID_SPEAKER_INDEX = 0
        self.FILE_PATH_INDEX = 1
        self.SPEAKER_GENDER_INDEX = 2
        self.VAD_POINTS_INDEX = 3

        self.MIN_AMP = 10000.0
        self.AMP_FAC = 10000.0
        self.THRESHOLD = 40

        self.data_base = data_base

        self.M = self.microphones_train # n microphones
        self.sample_rate = sample_rate  # sampling rate

        self.tf_converter = TFRecordsConverterDC()

        self.beamFormer = BF.BeamFormer(d=0.2, m=self.M)

        self.n_training = 0.8

        self.ids_speakers_in_train = []
        self.ids_speakers_in_test = []

        self.ids_speakers_in_train, self.ids_speakers_in_test = self.generate_ids()

    def get_random_audio(self, speaker_data):
        while True:
            random_audio_id = np.random.randint(0, len(speaker_data))
            random_audio_data = speaker_data[random_audio_id]
            if len(random_audio_data[self.VAD_POINTS_INDEX]) > 0:
                return random_audio_data

    def get_random_start(self, audio_signal, window_length):
        VAD_list = [int(f) for f in audio_signal[self.VAD_POINTS_INDEX].split(",")]
        VAD_random = random.randrange(0, len(VAD_list), 2)
        random_start = random.randrange(VAD_list[VAD_random], VAD_list[VAD_random + 1] - window_length)
        return random_start

    def get_random_speakers_data(self, ids_speakers, n):
        sources_ids = random.sample(ids_speakers, n)
        sources = [self.data_base[self.data_base[:, self.ID_SPEAKER_INDEX] == id_] for id_ in sources_ids]
        return sources

    def normalize_signal(self, s1):
        if s1.max() > 1.0:
            s1 = s1 * 1.0 / float(s1.max())
        if s1.min() < -1.0:
            s1 = s1 * (-1.0) / float(s1.min())
        return s1

    def get_random_speaker_audio(self, speaker_data, folder_path, window_length):
        random_audio_data = self.get_random_audio(speaker_data)
        VAD_random_start = self.get_random_start(random_audio_data, window_length)
        audio_path = folder_path + random_audio_data[1].strip()
        return random_audio_data, VAD_random_start, audio_path

    def to_train(self):
        # Vector of angles is created, from -90 to 90, steps of 45, and converting it to radians
        angles_vector = np.arange(-90, 90 + 45, 45) * math.pi / 180.0
        # The buffer is created, it will contain the files that are created while the gpu is busy
        tf_data_buffer = []
        # An observer is created to help us see what files of interest to us have already been deleted
        file_observer = fo.FileObserver(files=self.training_tf_records_files)
        # Infinite loop to create training files that have been deleted
        while True:
            # While loop if the buffer size limit is not exceeded
            while len(tf_data_buffer) < self.training_buffer_size_limit:
                # A list is created where tfrecords files are saved
                tf_data = []
                # The range of speakers contained in the training file is iterated.
                for speaker in tqdm(range(self.n_speakers_in_training_file)):
                    # The number of speakers that the mix will have is randomly selected
                    this_n_speakers_in_audio = np.random.randint(self.min_mixed_speakers_in_training_file,
                                                                 self.max_mixed_speakers_in_training_file)
                    #the information of a certain number of training announcers from the database is randomly selected
                    speakers_data = self.get_random_speakers_data(self.ids_speakers_in_train, this_n_speakers_in_audio)
                    #Each speaker can have one or more samples, therefore, one is chosen at random for each speaker, this is repeated according to the times that the user defines it
                    for n in range(self.n_repeated_speaker_in_training_file):
                        # Variable is created to keep the audio signal value of the speakers
                        audio_signals = np.zeros((this_n_speakers_in_audio, self.window_length))
                        # Variable is created to keep the energy value that audio signals have
                        speakers_audio_energies = np.zeros(this_n_speakers_in_audio)
                        # It iterates over each of the speakers, taking random audio
                        for x, speaker_data in enumerate(speakers_data):
                            # The audio data is obtained: start VAD and audio path
                            random_audio_data, VAD_random_start, audio_path = self.get_random_speaker_audio(
                                speaker_data,
                                self.audio_folder_path,
                                self.window_length)
                            # The signal audio is loaded
                            raw_signal, sr = sf.read(audio_path)
                            # It is cut in the part indicated by VAD
                            audio_signals[x] = raw_signal[VAD_random_start:VAD_random_start + self.window_length]
                            # The audio energy is calculated
                            speakers_audio_energies[x] = np.sqrt(
                                np.square(audio_signals[x]).sum() / float(len(audio_signals[x])))

                        # A noise scale is obtained from the highest energy value in the list
                        noise_scale = np.max(speakers_audio_energies) / speakers_audio_energies

                        # All values are normalized according to the noise scale
                        for h in range(this_n_speakers_in_audio):
                            audio_signals[h] = self.normalize_signal(audio_signals[h] * noise_scale[h])

                        # The speaker's angles are randomly obtained
                        doas = random.sample(angles_vector, this_n_speakers_in_audio)

                        # The first angle is considered as the one of interes (SOI)
                        doa_steer = doas[0]
                        
                        # A matrix of the size [microphones + 2, window size] is created (the last two data of the first dimension keep the signal of the
                        # source of interest of microphone 0 and source of interference of microphone 0,
                        # this because also it is given the same shift).

                        X = np.zeros([self.M + 2, self.window_length])

                        # All signals are mixed, this will be used to simulate microphone 0
                        mix_signal = np.sum(audio_signals[0:, :], axis=0) / float(this_n_speakers_in_audio)
                        
                        # A variable is created where the signals will be saved with offset to simulate the microphone signal 1
                        signal_delay = np.zeros(self.window_length)
                        for audio_source, doa in zip(audio_signals, doas):
                            signal_delay += delay_f(audio_source,
                                                    (self.beamFormer.d / self.beamFormer.c) * math.sin(doa), self.sample_rate)
                        # Data is normalized
                        signal_delay = signal_delay / float(this_n_speakers_in_audio)
                        # microphone 0
                        X[0, :] = mix_signal
                        # microphone 1
                        X[1, :] = signal_delay
                        # source of interest at microphone 0
                        X[2, :] = audio_signals[0, :]
                        # interference sources at microphone 0
                        X[3, :] = np.sum(audio_signals[1:, :], axis=0) / float(this_n_speakers_in_audio - 1)

                        beamformer_result = self.beamFormer.phase_mask(X=X,
                                                                       doa_steer=doa_steer,
                                                                       phase_diff_threshold=self.phase_diff_threshold,
                                                                       N=self.window_length,
                                                                       nframes=self.nframes, fs=self.sample_rate)

                        beamformer_result = beamformer_result[:, self.nframes * 4:self.nframes * -3]

                        # beamformer_result[0, :] SOI calculated by the beamformer
                        # beamformer_result[1, :] Interference calculated by the beamformer
                        # beamformer_result[2, :] audio at microphone 0
                        # beamformer_result[3, :] SOI (Clean)
                        # beamformer_result[4, :] Interference (Clean)

                        # stft are calculated for all resulting signals
                        _, _, Z_s1 = signal.stft(beamformer_result[3, :], fs=self.sample_rate, nperseg=self.nperseg,
                                                 noverlap=self.noverlap)
                        _, _, Z_s2 = signal.stft(beamformer_result[4, :], fs=self.sample_rate, nperseg=self.nperseg,
                                                 noverlap=self.noverlap)
                        _, _, Z_X_0 = signal.stft(beamformer_result[2, :], fs=self.sample_rate, nperseg=self.nperseg,
                                                  noverlap=self.noverlap)
                        _, _, Z_DOA = signal.stft(beamformer_result[0, :], fs=self.sample_rate, nperseg=self.nperseg,
                                                  noverlap=self.noverlap)
                        _, _, Z_interf = signal.stft(beamformer_result[1, :], fs=self.sample_rate, nperseg=self.nperseg,
                                                     noverlap=self.noverlap)
                        # the magnitudes are passed to decibels
                        db_mag_s1 = to_dB_mag(np.abs(Z_s1), self.MIN_AMP, self.AMP_FAC)
                        db_mag_s2 = to_dB_mag(np.abs(Z_s2), self.MIN_AMP, self.AMP_FAC)
                        db_mag_X_0 = to_dB_mag(np.abs(Z_X_0), self.MIN_AMP, self.AMP_FAC)
                        db_mag_DOA = to_dB_mag(np.abs(Z_DOA), self.MIN_AMP, self.AMP_FAC)
                        db_mag_interf = to_dB_mag(np.abs(Z_interf), self.MIN_AMP, self.AMP_FAC)

                        # The VAD frequency mask is created
                        max_mag = np.max(db_mag_X_0)
                        speech_VAD = (db_mag_X_0 > (max_mag - self.THRESHOLD)).astype(int)
                        # Masks are created from the beamformer
                        Y_beamformer = np.array([db_mag_DOA > db_mag_interf, db_mag_DOA < db_mag_interf]).astype(int)
                        # data is normalized
                        n_db_mag_X_0 = (db_mag_X_0 - db_mag_X_0.mean()) / float(db_mag_X_0.std())

                        # Source of interest and interference are calculated using the mask created by the beamformer
                        n_db_mag_source = n_db_mag_X_0 * Y_beamformer[0]
                        n_db_mag_interf = n_db_mag_X_0 * Y_beamformer[1]

                        # IBM is calculated
                        Y = np.array([db_mag_s1 > db_mag_s2, db_mag_s1 < db_mag_s2]).astype(int)
                        # _, source_recovered = signal.istft(Y[0] * Z_X_0 , fs=self.sample_rate ,nperseg= NPERSEG, noverlap = NOVERLAP)
                        # _, inter_recovered = signal.istft(Y[1] * Z_X_0 , fs=self.sample_rate ,nperseg= NPERSEG, noverlap = NOVERLAP)
                        Y = np.transpose(Y, [1, 2, 0])

                        # sf.write('PhaseMask_original_ref.wav', o_[3, :], self.sample_rate)
                        # sf.write('PhaseMask_original_inter.wav', o_[4, :], self.sample_rate)
                        # sf.write('PhaseMask_beamformer_ref.wav', o_[0, :], self.sample_rate)
                        # sf.write('PhaseMask_beamformer_inter.wav', o_[1, :], self.sample_rate)
                        # sf.write('PhaseMask_beamformer_m0.wav', o_[2, :], self.sample_rate)
                        # sf.write('PhaseMask_mask_ref.wav', source_recovered, self.sample_rate)
                        # sf.write('PhaseMask_mask_inter.wav', inter_recovered, self.sample_rate)
                        # exit(0)
                        complex_X_0 = np.array([Z_X_0.real, Z_X_0.imag])
                        complex_X_0 = np.transpose(complex_X_0, [1, 2, 0])
                        data = {
                            'n_db_mag_X_0': self.tf_converter.bytes_feature(n_db_mag_X_0.astype(np.float32).tostring()),
                            'n_db_mag_ref': self.tf_converter.bytes_feature(n_db_mag_source.astype(np.float32).tostring()),
                            'n_db_mag_interf': self.tf_converter.bytes_feature(
                                n_db_mag_interf.astype(np.float32).tostring()),
                            'complex_X_0': self.tf_converter.bytes_feature(complex_X_0.astype(np.float32).tostring()),
                            'MASK': self.tf_converter.bytes_feature(Y.astype(np.uint8).tostring()),
                            'VAD': self.tf_converter.bytes_feature(speech_VAD.astype(np.uint8).tostring())
                        }

                        tf_data.append(data)
                        if len(tf_data_buffer) > 0:  #if we have data in the buffer
                            if not file_observer.exist_all_files():  #and if there is no file
                                print("lvl 2")
                                name_file = file_observer.missing_file() # name of the missing file is obteined
                                self.tf_converter.set_features(tf_data_buffer.pop()).convert_to_tf(
                                    name_file) # the missing file is created

                tf_data_buffer.append(tf_data)   # created data is added to the buffer
                while not file_observer.exist_all_files():  # while files are missing
                    if len(tf_data_buffer) > 0:  #  If the buffer has data, the file is created
                        print("lvl 1")
                        name_file = file_observer.missing_file()
                        self.tf_converter.set_features(tf_data_buffer.pop()).convert_to_tf(name_file)
                    else:
                        break  ##else if it has nothing, break the loop

            file_observer.wait_if_exist_files()  # if the buffer is full, wait until a file is missing
            print("lvl 0")
            name_file = file_observer.missing_file()
            self.tf_converter.set_features(tf_data_buffer.pop()).convert_to_tf(name_file)

    def create_from_file(self,file):
        speakers_test_list = []
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                n_speakers_in_audio = int(row[0])
                speakers_list = [ audio for audio in row[1].split('@')]
                VAD_list = [[int (j) for j in i.split('-')] for i in row[2].split('@')]
                angles = [float(i) for i in row[3].split("@")]
                speakers_test_list.append([n_speakers_in_audio,speakers_list,VAD_list,angles])
        return speakers_test_list

    def create_file(self,speakers_test_list,file):
        file_txt = open(file,'w')
        file_string = ""
        for row in speakers_test_list:
            n_speakers_in_audio = str(row[0])
            speakers_list = '@'.join([str(i) for i in row[1]])
            VAD_list = '@'.join([ '{}-{}'.format(V[0],V[1]) for V in row[2]])#[[int (j) for j in i.split('-')] for i in row[2].split('@')]
            angles = '@'.join([ str(doa) for doa in row[3]])
            #speakers_test_list.append([n_speakers_in_audio,speakers_list,VAD_list,angles])
            file_string+= "{},{},{},{}\n".format(n_speakers_in_audio,speakers_list,VAD_list,angles)
        file_txt.write(file_string)
        file_txt.close()

    def create_random(self):
        angles_vector = np.arange(-90, 90 + 45, 45) * math.pi / 180.0
        speakers_test_list = []
        print("Creating random dataset")
        for speaker in tqdm(range(self.n_speakers_in_test_file)):
            this_n_speakers_in_audio = np.random.randint(self.min_mixed_speakers_in_test_file,
                                                         self.max_mixed_speakers_in_test_file)
            # Randomly select information from a certain number of training speakers in the database
            speakers_data = self.get_random_speakers_data(self.ids_speakers_in_test, this_n_speakers_in_audio)
            speakers_list = []
            VAD_list = []
            for x, speaker_data in enumerate(speakers_data):
                while True:
                    # The audio data is obtained: audio data and audio path
                    random_audio_data, _, audio_path = self.get_random_speaker_audio(
                        speaker_data,
                        self.audio_folder_path,
                        self.window_length)
                    # The audio signal is loaded
                    raw_signal, sr = sf.read(audio_path)
                    # it is validated that it is the size that is needed, otherwise it will select another signal
                    if len(random_audio_data[3]) > 0 and len(raw_signal) > self.window_length_test:
                        speakers_list.append(random_audio_data[1].strip())
                        break
                # the window is taken from the zero position
                VAD_list.append([0,self.window_length_test])
            angles = random.sample(angles_vector, this_n_speakers_in_audio)
            speakers_test_list.append([this_n_speakers_in_audio,speakers_list,VAD_list,angles])
        return speakers_test_list

    def to_test(self,configuration_test_file):
        tf_data = []
        eval_result = np.zeros((3,self.n_speakers_in_test_file))
        
        self.beamFormer_test = BF.BeamFormer(d=0.2, m=self.microphones_test)

        #checking paths
        if not os.path.exists(self.tf_records_test_path+"rec_audios/"):
            os.makedirs(self.tf_records_test_path+"rec_audios/")

        #it is checked if a file should be loaded
        if(configuration_test_file ==""):
            speakers_test_list = self.create_random()
            #create file from random data
            self.create_file(speakers_test_list,"data_set.csv")
        else:
        	#load file
            speakers_test_list = self.create_from_file(configuration_test_file) 
        
        
        for i in range(len(speakers_test_list)):
            speakers_test_list[i][1] = [ self.audio_folder_path + audio for audio in speakers_test_list[i][1]]

        speaker = -1
        for row in tqdm(speakers_test_list): 
            speaker+=1
            #row[0] number of mixed speakers
            #row[1] speaker files
            #row[2] VAD list
            #row[3] angles of speakers

            this_n_speakers_in_audio = row[0]
            speakers_list = row[1]
            VAD_list = row[2]
            angles = row[3]
            audio_signals = np.zeros((this_n_speakers_in_audio, self.window_length_test))
            speakers_audio_energies = np.zeros(this_n_speakers_in_audio)

            for x, audio_path in enumerate(speakers_list):
                raw_signal, sr = sf.read(audio_path)
                audio_signals[x] = raw_signal[VAD_list[x][0]:VAD_list[x][1]]
                speakers_audio_energies[x] = np.sqrt(np.square(audio_signals[x]).sum() / float(len(audio_signals[x])))

            noise_scale = np.max(speakers_audio_energies) / speakers_audio_energies

            # All values are normalized according to the noise scale
            for h in range(this_n_speakers_in_audio):
                audio_signals[h] = self.normalize_signal(audio_signals[h] * noise_scale[h])

            # The speaker's angles are randomly obtained
            doas = angles#random.sample(angles_vector, this_n_speakers_in_audio)

            # The first angle is considered as the one of interest (SOI)
            doa_steer = doas[0]
                        
            # A matrix of the size [microphones + 2, window size] is created (the last two data of the first dimension keep the signal of the
            # source of interest of microphone 0 and source of interference of microphone 0,
            # this because also it is given the same shift).
            X = np.zeros([self.M + 2, self.window_length_test])
            # All signals are mixed, this will be used to simulate microphone 0
            mix_signal = np.sum(audio_signals[0:, :], axis=0) / float(this_n_speakers_in_audio)
            # A variable is created where the signals with delay will be keep to simulate the microphone signal 1
            signal_delay = np.zeros(self.window_length_test)
            for audio_source, doa in zip(audio_signals, doas):
                signal_delay += delay_f(audio_source, (self.beamFormer.d / self.beamFormer.c) * math.sin(doa), self.sample_rate)
            # data is normalized
            signal_delay = signal_delay / float(this_n_speakers_in_audio)

            X[0, :] = mix_signal   # microphone 0

            X[1, :] = signal_delay   # microphone 1
            # SOI microphone 0
            X[2, :] = audio_signals[0, :]
            # source of interference in microphone 0
            X[3, :] = np.sum(audio_signals[1:, :], axis=0) / float(this_n_speakers_in_audio - 1)

            beamformer_result = self.beamFormer_test.phase_mask(X=X,
                                                           doa_steer=doa_steer,
                                                           phase_diff_threshold=self.phase_diff_threshold,
                                                           N=self.window_length_test,
                                                           nframes=self.nframes, fs=self.sample_rate)

            beamformer_result = beamformer_result[:, self.nframes * 4:self.nframes * -3]

            # beamformer_result[0, :] SOI calculated by the beamformer
            # beamformer_result[1, :] Interference calculated by the beamformer
            # beamformer_result[2, :] audio at microphone 0
            # beamformer_result[3, :] SOI (Clean)
            # beamformer_result[4, :] Interference (Clean)

            beam_source = beamformer_result[0, :]
            beam_inter = beamformer_result[1, :]
            original_source = beamformer_result[3, :]
            original_inter = beamformer_result[4, :]
            m0_ = beamformer_result[2, :]

            # Evaluation Beamformer

            (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(beamformer_result[3:5, :], beamformer_result[0:2, :])
            eval_result[0,speaker] = sdr[perm.tolist().index(0)]
            eval_result[1,speaker] = sir[perm.tolist().index(0)]
            eval_result[2,speaker] = sar[perm.tolist().index(0)]

            #Beamformer results
            sf.write(self.tf_records_test_path+"rec_audios/{}_a_beam.wav".format(speaker),beam_source, self.sample_rate)
            sf.write(self.tf_records_test_path+"rec_audios/{}_b_beam.wav".format(speaker),beam_inter, self.sample_rate)
            
            for i in range(self.n_files_per_speaker_in_training_file):
                # A frame of the source of interest obtained with the beamformer is obtained
                source = beam_source[self.window_size * i:self.window_size * i + self.window_size] + np.finfo(np.float).eps
                # A frame of the interference source obtained with the beamformer is obtained
                interf = beam_inter[self.window_size * i:self.window_size * i + self.window_size] + np.finfo(
                    np.float).eps
                # A frame is obtained from the source of interest of the microphone 0
                s1 = original_source[self.window_size * i:self.window_size * i + self.window_size] + np.finfo(np.float).eps
                # A frame is obtained from the original interference source of the microphone 0
                s2 = original_inter[self.window_size * i:self.window_size * i + self.window_size] + np.finfo(
                    np.float).eps
                # An audio frame is obtained from the microphone 0
                m0 = m0_[self.window_size * i:self.window_size * i + self.window_size] + np.finfo(np.float).eps

                # stft are calculated for all resulting signals
                _, _, Z_s1 = signal.stft(s1, fs=self.sample_rate, nperseg=self.nperseg, noverlap=self.noverlap)
                _, _, Z_s2 = signal.stft(s2, fs=self.sample_rate, nperseg=self.nperseg, noverlap=self.noverlap)
                _, _, Z_X_0 = signal.stft(m0, fs=self.sample_rate, nperseg=self.nperseg, noverlap=self.noverlap)
                _, _, Z_source = signal.stft(source, fs=self.sample_rate, nperseg=self.nperseg, noverlap=self.noverlap)
                _, _, Z_interf = signal.stft(interf, fs=self.sample_rate, nperseg=self.nperseg, noverlap=self.noverlap)

                # the magnitudes are passed to decibels
                db_mag_s1 = to_dB_mag(np.abs(Z_s1), self.MIN_AMP, self.AMP_FAC)
                db_mag_s2 = to_dB_mag(np.abs(Z_s2), self.MIN_AMP, self.AMP_FAC)
                db_mag_X_0 = to_dB_mag(np.abs(Z_X_0), self.MIN_AMP, self.AMP_FAC)
                db_mag_source = to_dB_mag(np.abs(Z_source), self.MIN_AMP, self.AMP_FAC)
                db_mag_interf = to_dB_mag(np.abs(Z_interf), self.MIN_AMP, self.AMP_FAC)

                # The VAD frequency mask is created
                max_mag = np.max(db_mag_X_0)
                speech_VAD = (db_mag_X_0 > (max_mag - self.THRESHOLD))  # .astype(int)

                Y_beamformer = np.array([db_mag_source > db_mag_interf, db_mag_source < db_mag_interf]).astype(int)

                n_db_mag_X_0 = (db_mag_X_0 - db_mag_X_0.mean()) / float(db_mag_X_0.std())
                n_db_mag_source = n_db_mag_X_0 * Y_beamformer[0]
                n_db_mag_interf = n_db_mag_X_0 * Y_beamformer[1]
                # IBM is calculated
                Y = np.array([db_mag_s1 > db_mag_s2, db_mag_s1 < db_mag_s2]).astype(int)
                # _, source_recovered = signal.istft(Y[0] * Z_X_0 , fs=self.sample_rate ,nperseg= NPERSEG, noverlap = NOVERLAP)
                # _, inter_recovered = signal.istft(Y[1] * Z_X_0 , fs=self.sample_rate ,nperseg= NPERSEG, noverlap = NOVERLAP)
                Y = np.transpose(Y, [1, 2, 0])

                if Z_X_0.shape != (self.input_size, self.time_step):
                    print("ERROR")

                if db_mag_source.shape != (self.input_size, self.time_step):
                    print("ERROR")

                # print(db_mag_source.shape)
                # exit(0)
                # sf.write('PhaseMask_original_ref.wav', s1, self.sample_rate)
                # sf.write('PhaseMask_original_inter.wav', s2, self.sample_rate)
                # sf.write('PhaseMask_beamformer_ref.wav', source, self.sample_rate)
                # sf.write('PhaseMask_beamformer_inter.wav', interf, self.sample_rate)
                # sf.write('PhaseMask_beamformer_m0.wav', m0, self.sample_rate)
                # sf.write('PhaseMask_mask_ref.wav', source_recovered, self.sample_rate)
                # sf.write('PhaseMask_mask_inter.wav', inter_recovered, self.sample_rate)
                # exit(0)
                complex_X_0 = np.array([Z_X_0.real, Z_X_0.imag])
                complex_X_0 = np.transpose(complex_X_0, [1, 2, 0])
                data = {
                    'n_db_mag_X_0': self.tf_converter.bytes_feature(n_db_mag_X_0.astype(np.float32).tostring()),
                    'n_db_mag_ref': self.tf_converter.bytes_feature(n_db_mag_source.astype(np.float32).tostring()),
                    'n_db_mag_interf': self.tf_converter.bytes_feature(n_db_mag_interf.astype(np.float32).tostring()),
                    'complex_X_0': self.tf_converter.bytes_feature(complex_X_0.astype(np.float32).tostring()),
                    'MASK': self.tf_converter.bytes_feature(Y.astype(np.uint8).tostring()),
                    'VAD': self.tf_converter.bytes_feature(speech_VAD.astype(np.uint8).tostring())
                }

                tf_data.append(data)
        file = open(self.tf_records_test_path + 'sdr_sir_sar_summary.csv',"w")
        file.write('audio_id,sdr,sir,sar\n')
        for i in range(eval_result.shape[1]):
            file.write('{},{},{},{}\n'.format(i,eval_result[0,i],eval_result[1,i],eval_result[2,i]))
        file.close()
        self.tf_converter.set_features(tf_data).convert_to_tf(self.tf_records_test_path + '0.tfrecords')


    def generate_ids(self):
        # for woman and man

        self.dataBase_A = self.data_base[self.data_base[:, self.SPEAKER_GENDER_INDEX] == "F"]
        self.dataBase_B = self.data_base[self.data_base[:, self.SPEAKER_GENDER_INDEX] == "M"]

        ids_speakers = np.unique(self.data_base[:, self.ID_SPEAKER_INDEX])
        ids_speakers_f = np.unique(self.dataBase_A[:, self.ID_SPEAKER_INDEX])
        ids_speakers_m = np.unique(self.dataBase_B[:, self.ID_SPEAKER_INDEX])

        print("Speakers :", ids_speakers.shape)
        print("Woman:", ids_speakers_f.shape)
        print("Man:", ids_speakers_m.shape)

        ids_speakers_train = ids_speakers[0:int(math.floor(round(ids_speakers.shape[0] * self.n_training)))]
        ids_speakers_test = ids_speakers[int(math.ceil(round(ids_speakers.shape[0] * self.n_training)))::]

        print("Evaluation type: mix")
        print("Training:", len(ids_speakers_train))
        print("Test:", len(ids_speakers_test))

        return ids_speakers_train, ids_speakers_test


def main():
    configuration_file = str(sys.argv[1])
    if configuration_file == "":
        print("ERROR: you need to define param: configuration <configuration_file>.json ")
        exit(0)

    PARAMS = None

    with open(configuration_file, 'r') as f:
        f = f.read()
        PARAMS = json.loads(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

    print(vars(PARAMS))

    audio_folder_path = PARAMS.PATHS.database_folder_path  # "/home/ar/source_separation/LibriSpeech_6000/"

    # First, load the database
    data_base = np.load(PARAMS.PATHS.npy_database_path)
    # Indicates the name of the first file generated by the process [0,1,2,3] depends on the number of process you have.
    files_ = int(sys.argv[2])
    # Always create 8 tfrecords files
    files_total = int(sys.argv[3])
    # Number of process you will have running for the experiment (even, 2,4,8,10)
    processes = int(sys.argv[4])

    configuration_test_file = ""
    if(len(sys.argv) > 5):
        configuration_test_file = sys.argv[5]

    print(files_total / float(processes))
    print(files_ + int(files_total / float(processes)))
    tf_records_training_path = PARAMS.PATHS.tf_records_training_path  # str(sys.argv[6])
    training_tf_records_files = [tf_records_training_path + str(k) + ".tfrecords" for k in
                                 range(files_, files_ + int(files_total / float(processes)))]

    print(training_tf_records_files)

    sample_audio_signal = data_base[0]

    audio_path = audio_folder_path + sample_audio_signal[1].strip()
    _, sr = sf.read(audio_path)

    if PARAMS.DATA_GENERATOR.sample_rate != sr:
        print("ERROR: different sample rate in database")
        exit(0)

    generateDatabase = GenerateDatabase(data_base=data_base,
                                        training_tf_records_files=training_tf_records_files,
                                        n_speakers_in_training_file=PARAMS.DATA_GENERATOR.n_speakers_train,
                                        n_repeated_speaker_in_training_file=PARAMS.DATA_GENERATOR.n_repeated_speakers_train,
                                        max_mixed_speakers_in_training_file=PARAMS.DATA_GENERATOR.max_speakers_train,
                                        min_mixed_speakers_in_training_file=PARAMS.DATA_GENERATOR.min_speakers_train,
                                        tf_records_test_path=PARAMS.PATHS.tf_records_test_path,
                                        n_speakers_in_test_file=PARAMS.DATA_GENERATOR.n_speakers_test,
                                        n_files_per_speaker_in_training_file=PARAMS.DATA_GENERATOR.files_per_speaker_test,
                                        max_mixed_speakers_in_test_file=PARAMS.DATA_GENERATOR.max_speakers_test,
                                        min_mixed_speakers_in_test_file=PARAMS.DATA_GENERATOR.min_speakers_test,
                                        phase_diff_threshold=PARAMS.DATA_GENERATOR.phase_diff_threshold,
                                        training_buffer_size=PARAMS.DATA_GENERATOR.training_buffer_size,
                                        nframes=PARAMS.DATA_GENERATOR.nframes,
                                        nperseg=PARAMS.DATA_GENERATOR.stft_nperseg,
                                        noverlap=PARAMS.DATA_GENERATOR.stft_noverlap,
                                        window_size=PARAMS.DATA_GENERATOR.stft_window,
                                        audio_folder_path=PARAMS.PATHS.database_folder_path,
                                        sample_rate=PARAMS.DATA_GENERATOR.sample_rate,
                                        microphones_train=PARAMS.DATA_GENERATOR.microphones_train,
                                        microphones_test = PARAMS.DATA_GENERATOR.microphones_test)

    if files_ == 0:
        if not os.path.exists(PARAMS.PATHS.tf_records_test_path + '0.tfrecords'):
            print("Creating test ... ")
            generateDatabase.to_test(configuration_test_file)
    generateDatabase.to_train()


if __name__ == "__main__":
    main()
