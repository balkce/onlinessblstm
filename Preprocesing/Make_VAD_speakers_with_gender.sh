#!/bin/bash
# window_size /home/ar/source_separation/LibriSpeech/ 
python Preprocesing/Make_VAD_speakers_with_gender.py 0 18586 0 $1 speakers_with_gender_500.npy $2 &
python Preprocesing/Make_VAD_speakers_with_gender.py 18586 37172 1 $1 speakers_with_gender_500.npy $2 &
python Preprocesing/Make_VAD_speakers_with_gender.py 37172 55758 2 $1 speakers_with_gender_500.npy $2 &
python Preprocesing/Make_VAD_speakers_with_gender.py 55758 74344 3 $1 speakers_with_gender_500.npy $2 &
python Preprocesing/Make_VAD_speakers_with_gender.py 74344 92930 4 $1 speakers_with_gender_500.npy $2 &
python Preprocesing/Make_VAD_speakers_with_gender.py 92930 111516 5 $1 speakers_with_gender_500.npy $2 &
python Preprocesing/Make_VAD_speakers_with_gender.py 111516 130102 6 $1 speakers_with_gender_500.npy $2 &
python Preprocesing/Make_VAD_speakers_with_gender.py 130102 148688 7 $1 speakers_with_gender_500.npy $2 