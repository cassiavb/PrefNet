# PrefNet

Go to a suitable location and clone repository:

```
git clone https://github.com/cassiavb/PrefNet.git
cd PrefNet
```

## Installation of Python dependencies with virtual environment

Make a directory to house virtual environments if you don't already have one, and move to it:

```
virtualenv --distribute --python=/usr/bin/python3.6 env
source env/bin/activate
```

With the virtual environment activated, you can now install the necessary packages.

```
pip install --upgrade pip
pip install -r ./requirements.txt
```

## Pre-trained models

Pre-trained models can be found in [models/](models/)

## Predict preference from two wavefiles using a pre-trained model

```
./util/submit_tf.sh scripts/predict.py -c config/example.cfg -i1 file1.wav -i2 file2.wav
```

## Train new model

```
./util/submit_tf.sh scripts/train.py -c config/example.cfg
```

## Calculate model's accuracy on testset

```
./util/submit_tf.sh scripts/test.py -c config/example.cfg
```

## Prepare data for training and testing

Data from Exp1-7 in the expected format can be found [here](http://data.cstr.ed.ac.uk/cvbotinh/SM/Y2/data_dir.zip).

### Wavfiles 

Should be located in (all files should be sampled at 16kHz):

data_dir/processed/wav16k/

all files should be named following: EXXX___sentence___system.wav  

XXX is a unique number that identifies the experiment (listening test).

### Score files (one file per experiment) 

Should be located in:

data_dir/processed/scores/EXXX_scores.float

The score file is a binary file in float format that contains a matrix M with pair wise scores between 0 and 1:  
M(i,j) = preference of filename(j) over filename(i)  
If a pair was not compared in the listening test that matrix element is set to -1  
For an experiment with 100 wavfiles M should be a 100x100 matrix. 

### Filename and pair files (one file per experiment) 

Should be located in:

data_dir/processed/filenames/EXXX_filenames.txt  
data_dir/processed/filenames/EXXX_pairs.txt 

The filenames file is an ascii file containing the name of the wavfiles indexed in the matrix. For this example experiment this should have 100 lines. 

The pairs file is an ascii fike containing all pairs of wavfiles that were compared in the experiment. The number of lines should match the number of elements in M that are different than -1.  

