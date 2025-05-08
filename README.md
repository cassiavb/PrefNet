# PrefNet

This repository contains code, pre-trained models and data for the following Interspeech 2022 paper:

["Predicting pairwise preferences between TTS audio stimuli using parallel ratings data and anti-symmetric twin neural networks"](https://www.isca-archive.org/interspeech_2022/valentinibotinhao22_interspeech.pdf)  
Cassia Valentini-Botinhao, Manuel Sam Ribeiro, Oliver Watts, Korin Richmond, Gustav Eje Henter

## Installation

Go to a suitable location and clone repository:

```
git clone https://github.com/cassiavb/PrefNet.git
cd PrefNet/
```

Make a directory to house virtual environments if you don't already have one, and move to it:

```
virtualenv --distribute --python=/usr/bin/python3.6 env
source env/bin/activate
```

With the virtual environment activated, you can now install the necessary packages.

```
pip install --upgrade pip
```

To replicate env used in IS22 subimssion:
```
pip install tensorflow-gpu==1.10.1
pip install Soundfile==0.10.2
pip install numpy==1.19.4
pip install Keras==2.2.2
pip install kapre==0.1.3.1
pip install h5py==2.10.0
```

## Pre-trained models

Pre-trained models can be found in [models/](models/)

## Preference prediction

Predict the preference between two wavefiles using pre-trained models (GPU only):
```
./util/submit_tf.sh scripts/predict.py -c config/GRU_4.cfg -m models/interspeech_2022/LargeSet.hdf5 -i1 file1.wav -i2 file2.wav
```

## Data

Data from Eval1-7 in the expected format can be found [here](http://data.cstr.ed.ac.uk/cvbotinh/SM/Y2/data_dir.zip).

Data from Large Set coming soon!

## Train model

```
./util/submit_tf.sh scripts/train.py -c config/GRU_4.cfg
```

## Evaluate model

On test set:
```
./util/submit_tf.sh scripts/test.py -c config/GRU_4.cfg
```

On cross validation test sets:
```
./util/submit_tf.sh scripts/test_crossval.py -c config/GRU_4.cfg
```

## Data preparation

### Wavfiles 

Should be located in:

data_dir/processed/wav16k/

should be named: EXXX___sentence___system.wav  

XXX is a unique number that identifies the evaluation (listening test).

### Score files (one file per evaluation) 

Should be located in:

data_dir/processed/scores/EXXX_scores.float

The score file is a binary file in float format that contains a matrix M with pair wise scores between 0 and 1:  
M(i,j) = preference of filename(j) over filename(i)  
If a pair was not compared in the listening test that matrix element is set to -1  
For an evaluation with 100 wavfiles M should be a 100x100 matrix. 

### Filename and pair files (one file per evaluation) 

Should be located in:

data_dir/processed/filenames/EXXX_filenames.txt  
data_dir/processed/filenames/EXXX_pairs.txt 

The filenames file is an ascii file containing the name of the wavfiles indexed in the matrix. For this example evaluation this should have 100 lines. 

The pairs file is an ascii fike containing all pairs of wavfiles that were compared in the evaluation. The number of lines should match the number of elements in M that are different than -1.  

