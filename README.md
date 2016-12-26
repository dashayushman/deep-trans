![alt tag](https://sigvoiced.files.wordpress.com/2016/09/deeptrans.png)
-----------------
**DeepTrans** is a character level language model for [transliterating](https://en.wikipedia.org/wiki/Transliteration) English text into Hindi. It is based on the attention mechanism presented in [[1]](http://arxiv.org/abs/1409.3215) and its implementation in [Tensorflow's](https://www.tensorflow.org/) [Sequence to Sequence](https://www.tensorflow.org/versions/r0.10/tutorials/seq2seq/index.html) models. This project has been inspired by the translation model presented in tensorflow's sequence to sequence model. This project comes with a pretrained model (2 layers with 256 units each) for Hindi but can be easily trained over the existing model or from scratch. The pretrained models are trained on lowercase words. If you wish to train your own model then feel free to do whatever you want and I would be glad if you could share your results and models with me. I hope to see interesting results.

#Prerequisites
1. [Tensorflow](https://www.tensorflow.org/) (Version >= 0.9)
2. Python 2.7

**I have tested it on an Ubuntu 15.04 with NVIDIA GeForce GT 740M Graphics card with Tensorflow running in a virtual environment. It should ideally run smoothly on any other system with tensorflow installed in it.**

#Installation and Setup
###Clone Repository
```
git clone https://github.com/dashayushman/deep-trans.git
```

###Run Tests
```
python transliterate.py --self_test
```
This will generate a fake model (2 layers 32 units per layer) with fake data and trains it for 5 steps.</br>
**If the code returns without any errors, proceed to the next step.**

###Download The Model and Vocabulary
1. Download the **pre-trained model** from [here](https://drive.google.com/open?id=0B39jMq4OCmFDcDFZbmdLNjNnTVU) and extract the model files to any folder in your system.
The folder structure for models looks something like the following,
```
trained_model
    |_version_1.0
            |_model_12_09_2016.zip
            |_model_12_09_2016.tar
    |_version_0.1
            |_model_9_08_2016.zip
            |_model_9_08_2016.tar
```
2. Download the **vocabulary** from [here](https://drive.google.com/open?id=0B39jMq4OCmFDSmh6cllVU2VyVXc) and extract the vocabulary files to any folder in your system.
The folder structure for vocabulary looks something like the following,
```
vocabulary
    |_version_1.0
            |_vocab_12_09_2016.zip
            |_vocab_12_09_2016.tar
    |_version_0.1
            |_vocab_9_08_2016.zip
            |_vocab_9_08_2016.tar
```

***The pretrained models and vocabularies are versioned with a date attached to the name of the compressed files. Downloading the latest version is recommended. You will find both .tar and .zip files in the download link. Both of them have the same model so you can download any one. Make sure that your mode and vocabulary date and version match.***</br></br>

#Load and Run
###Loading the model
Execute the following command from your commandline to load the pre-trained models and enter an interactive mode where you can input english strings in the standard input and check results there itself.
```
python transliterate.py --data_dir <path_to_vocabulary_directory> --train_dir <path_to_models_directory> --decode
```
Your commandline should have something like this
![alt tag](https://sigvoiced.files.wordpress.com/2016/09/yes.png)

You can enter your ***'English word'*** after the ***'>'*** in the command like and hit enter to see results.

###Transliterate a file
Execute the following command from your commandline to load the pre-trained models and transliterate an entire file.</br>
Make sure your file contains one english word per line and is named ***'test.en'***
```
python transliterate.py --data_dir <path_to_vocabulary_directory> --train_dir <path_to_models_directory> --transliterate_file --transliterate_file_dir <path_to_directory_that_contains_test.en>
```
If you get a ***'done generating the output file!!!'*** message on your commandline, then you are good to go. You will find a ***'results.txt'*** file in your ***'transliterate_file_dir'***

#Train Your Own Model
###Requirements
1. **Training and development files:** You will need two set of files for training your own model.
  * **Training Files:** You would need two training files with file names ***'train.rel.2.en'*** and ***'train.rel.2.hn'***. The ***'train.rel.2.en'*** should contain all the english words for training with one word per line and each character separated by a space. Similarly ***'train.rel.2.hn'*** should contain corresponding hindi words for the english words in ***'train.rel.2.en'*** with one word per line and each character separated by a space. Make sure that the English and Hindi words correspond otherwise you will end up training a very messy model.
  * **Development Files:** You would need two development files with file names ***'test.rel.2.en'*** and ***'test.rel.2.hn'***. The ***'test.rel.2.en'*** should contain english words for validation with one word per line and each character separated by a space. Similarly ***'test.rel.2.hn'*** should contain corresponding hindi words for the english words in ***'train.rel.2.en'*** with one word per line and each character separated by a space. Make sure that the English and Hindi words correspond.
2. Try not to overlap the development set and training set.
3. Keep these files in a directory.
4. **Very Important Point To Note:** Due to the Character encoding issues in python 2.7 I have to put these restrictions on formatting the data (adding spaces between every character in a word). I will soon release another version with Python3+ support and solve this encoding issue and remove this weird data formatting restriction.
5. This is how the data files should look like:</br>
![alt tag](https://sigvoiced.files.wordpress.com/2016/09/train.png)

###Training
Once you have the above files in a directory, execute the following command to start training your own model.
```
python transliterate.py --data_dir <path_to_directory_with_training_and_development_files> --train_dir <path_to_a_directory_to_save_checkpoints> ----size=2<number_units_per_layer> --num_layers=<number_of_layers> --steps_per_checkpoint=<number_of_steps_to_save_a_checkpoint>
```
The following is a real example of the above,
```
python transliterate.py --data_dir /home/ayushman/projects/transliterate/train_test_data/ --train_dir /home/ayushman/projects/transliterate/chkpnts/ --size=1024 --num_layers=5 --steps_per_checkpoint=1000
```

###FLAGS
The following is a list of available flags that you can set for changing the model parameters.

FLAG | VALUE TYPE | DEFAULT VALUE | DESCRIPTION
--- | --- | --- | ---
learning_rate | Float | 0.001 | Learning rate for backpropagation through time. |
learning_rate_decay_factor | Float | 0.99 | Learning rate decays by this much.
max_gradient_norm | Float | 5.0 | Clip gradients to this norm.
batch_size | Integer | 10 | Batch size to use during training.
size | Integer | 256 | Size of each model layer.
num_layers | Integer | 2 | Number of layers in the model.
en_vocab_size | Integer | 40000 | English vocabulary size.
hn_vocab_size | Integer | 40000 | Hindi vocabulary size.
data_dir | String(path) | /tmp | Data directory
transliterate_file_dir | String(path) | /tmp | Data directory
train_dir | String(path) | /tmp | Training directory (to save checkpoints or models).
max_train_data_size | Integer | 0 | Limit on the size of training data (0: no limit).
steps_per_checkpoint | Integer | 200 | How many training steps to do per checkpoint.
decode | Boolean | False | et to True for interactive decoding.
transliterate_file | Boolean | False | Set to True for transliterating a file.
self_test | Boolean | False | Run a self-test if this is set to True.


#References
1. [Sequence to Sequence Learning with Neural Networks](http://arxiv.org/abs/1409.3215)
