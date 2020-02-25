#!/usr/bin/env python
# -*- coding: utf-8 -*-
import soundfile as sf
import numpy as np
import VAD as vad
import sys
import os

datos = []

start = int(sys.argv[1])
end = int(sys.argv[2])

file_i = int(sys.argv[3]) #number of output file [0,1,2,...]
vad_window_size = int(sys.argv[4]) #VAD window length
data_file = str(sys.argv[5]) #file with spekears and genders
path_file = str(sys.argv[6]) #audio path

train_other_500_data = np.load(data_file)
print len(train_other_500_data)
x = 0
for d in train_other_500_data[start:end]:
    print x
    
    file = path_file+d[1].strip() # file path

    data, sr = sf.read(file)
    s0 , s1, s2 = vad.vad_analysis(data,sr,vad_window_size)
    datos.append([d[0],d[1].strip(),d[2],','.join([str(s) for s in s2.tolist()])]) # id, file, gender, VAD points [start,end,...,start,end]
    
    x+=1

if not os.path.exists('VAD_'+str(vad_window_size)):
	os.makedirs('VAD_'+str(vad_window_size))

np.save('VAD_'+str(vad_window_size)+"/VAD_speakers_with_gender_500_{}_{}.npy".format(vad_window_size,file_i),datos)
