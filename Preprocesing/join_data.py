import numpy as np
import sys
import os
base_path = "VAD_WINDOW_SIZE/VAD_speakers_with_gender_500_WINDOW_SIZE_{}.npy".replace("WINDOW_SIZE",str(sys.argv[1])) # "/home/ar/source_separation/VAD_32768/VAD_speakers_with_gender_500_VAD_WINDOW_SIZE_{}.npy" 
files = int(sys.argv[2]) # number of files

data = []
for i in range(files):
	if not os.path.exists(base_path.format(i)):
		print("Error file {} does not exist".format(base_path.format(i)))
		exit(0)

	data.append(np.load(base_path.format(i)))
dataBase = np.concatenate(data,axis=0)

np.save(base_path.replace("_{}.npy",".npy"),dataBase)