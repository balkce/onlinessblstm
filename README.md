# onlinessblstm
Lightweight online separation of the sound source of interest through a hybrid approach: 1) a phase-based frequency masking beaformer, and 2) BLSTM-based binary TF masking.

To run these experiments you need to follow three steps:

1. Create an environment (conda)
2. Prepare data
3. Run experiments


# 1. Create an environment:
```
conda env create -f Preprocesing/tf_1_10_gpu.yml
conda activate tf_1_10
```
# 2. Prepare data:

(Download or clone this repo and then run all the commands inside the main folder)
 
1.- Download the dataset:
```
wget http://www.openslr.org/resources/12/train-other-500.tar.gz
tar xvzf train-other-500.tar.gz
```
2.- Remove some text to make a csv file with speakers information.
```
tail -n +13 LibriSpeech/SPEAKERS.TXT > LibriSpeech/SPEAKERS.csv
```
3.- Make the speakers_with_gender_500.npy file containing information about speakers.
(file speakers_with_gender_500.npy will be generated in the current directory)
```
python Preprocesing/Make_speakers_with_gender.py LibriSpeech/SPEAKERS.csv Preprocesing/speakers_files_original.txt
```
4.- Then, run the next commands to create a file containing VAD of window size 32768 for ~1 second (window size 16384, note that we need twice the window size).
```
bash Preprocesing/Make_VAD_speakers_with_gender.sh 32768 ./
python Preprocesing/join_data.py 32768 8
```
5.- To create a file containing VAD of window length 16384 for ~0.5 seconds (window length 8192, note that we need twice the window size)
```
bash Preprocesing/Make_VAD_speakers_with_gender.sh 16384 ./
python Preprocesing/join_data.py 16384 8
```


# 3. Run experiments.

There are two important files to run the experiments, CreateDatabase.py to create the data and main.py to train and evaluate a model.

### Create data for train and evaluate ###

python CreateDatabase.py [configuration file] [id first file] [process] [num of files (at the moment always 8)] [optional csv file containing pre-selected speakers and audios]

For example, to run the experiment with configuration Exp0.json, you need generate data, in this case with 8 process: 

(Open one tmux session for command is recommended)
```
python CreateDataBase.py Exp0.json 0 8 8 Test_100.csv
python CreateDataBase.py Exp0.json 1 8 8 
python CreateDataBase.py Exp0.json 2 8 8 
python CreateDataBase.py Exp0.json 3 8 8 
python CreateDataBase.py Exp0.json 4 8 8 
python CreateDataBase.py Exp0.json 5 8 8 
python CreateDataBase.py Exp0.json 6 8 8 
python CreateDataBase.py Exp0.json 7 8 8 
```
Or you can run:
```
bash CreateDatabase.sh Exp0.json Test_100.csv
```
CreateDatabase.sh file contains the next instruction to be executed:
```
tmux new-session -d -s 0 'python CreateDataBase.py $1 0 8 8 $2'
tmux new-session -d -s 1 'python CreateDataBase.py $1 1 8 8'
tmux new-session -d -s 2 'python CreateDataBase.py $1 2 8 8'
tmux new-session -d -s 3 'python CreateDataBase.py $1 3 8 8'
tmux new-session -d -s 4 'python CreateDataBase.py $1 4 8 8'
tmux new-session -d -s 5 'python CreateDataBase.py $1 5 8 8'
tmux new-session -d -s 6 'python CreateDataBase.py $1 6 8 8'
tmux new-session -d -s 7 'python CreateDataBase.py $1 7 8 8'
```
to stop process open each tmux session [from 0 to 7] and make exit:
```
tmux a -t [session-id]
[ctrl+c]
exit
```
Important notes:
1. The processes (CreateDataBase.py) must be working whenever you are training, these processes are generating data using the CPU, while the gpu is used to train.
2. First time you create test file, ISR, SAR and SDR evaluation is calculated using the phase beamformer, those results are saved in [folder_test_name]/sdr_sir_sar_summary.csv


### Training a model ###

python main.py [json configuration file] [GPU ID] [load_session (true/false)] [id_experiment (folder name, if load session)]

For example, to run the experiment with configuration Exp0.json:
```
python main.py Exp0.json 0 false
```

Note: The folder name or ID is given by the main.py in training mode.

To run the experiments, you can use different configurations of parameters, the file Configurations_summary.csv describes in general the configuration of each json file.

### Evaluating a model ### 

To evaluate, assuming that the id of the experiment is 1 and that the model was fully trained (according to the configuration file parameters), the correct command to evaluate is:
```
python main.py Exp0.json 0 true 1
```
the same command works to continue training (if the process stopped before completing all global steps)

Test_100.csv file contains the information for the SIR, SAR, SDR evaluation. To replicate our experiments, make sure you use the Test_100.csv file, if you do not indicate a csv file, one csv will be generated with random speakers from the test set, and will be named default.csv.

Test_100.csv columns are separated by comma, and contains:
- column 0: number of mixed speakers
- column 1: speaker files separated by @
- column 2: VAD list separated by @, star and end separated by -
- column 3: angles of speakers separated by @

The results are saved in [main_path of the json configuration file]/results/[experiment id] /
There are two types of file results, condensed and detailed, the condensed results file contains only model information and the total average of SIR, SAR and SDR.

The detailed results file indicates the SIR, SAR and SDR for each audio.

Important note:
At the moment, in the Test_100.csv file, only 100 speakers are considered for the test, each speaker with ~ 10 seconds of audio (~ 1000 seconds of audio, ~ 16 minutes).
