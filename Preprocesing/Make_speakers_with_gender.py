import csv
import numpy as np
import sys

speakers_file_csv = str(sys.argv[1]) # SPEAKERS.csv PATH
speakers_file_txt = str(sys.argv[2]) # speakers_files.txt PATH



speakers_file_csv = open(speakers_file_csv,"r")
csv_reader = csv.reader(speakers_file_csv, delimiter = "|")
data_speakers = {}
for row in csv_reader:
	data_speakers[row[0].strip()] = row[1].strip() # id, gender
#find LibriSpeech -name \*.flac -print | grep 'train-other-500' > speakers_files.txt
train_other_500_files = open(speakers_file_txt)

train_other_500_data = []
for f in train_other_500_files:
	f_split = f.split("/")
	user_id = f_split[-3]
	gender = data_speakers[user_id] #user id
	train_other_500_data.append([user_id,f,gender])

print len(train_other_500_data)
#np.save("datos_con_genero_500.npy",np.array(new_data))
np.save("speakers_with_gender_500.npy",np.array(train_other_500_data))