![alt tag](https://sigvoiced.files.wordpress.com/2016/09/deeptrans.png)
-----------------
# DeepTrans
**DeepTrans** is a character level language model for [transliterating](https://en.wikipedia.org/wiki/Transliteration) English text into Hindi. It is based on the attention mechanism presented in [[1]](http://arxiv.org/abs/1409.3215) and its implementation in [Tensorflow's](https://www.tensorflow.org/) [Sequence to Sequence](https://www.tensorflow.org/versions/r0.10/tutorials/seq2seq/index.html) models. This project has been inspired by the translation model presented in tensorflow's sequence to sequence model. This project comes with a pretrained model for Hindi but can be easily trained over the existing model or from scratch.

#Prerequisites
1. [Tensorflow](https://www.tensorflow.org/) (Version >= 0.9)

**I have tested it on an Ubuntu 15.04 with NVIDIA GeForce GT 740M Graphics card with Tensorflow running in a virtual environment. It should ideally run smoothly on any other system with tensorflow installed in it.**

#Installation
###Clone Repository
```
npm install
```

#References
1. [Sequence to Sequence Learning with Neural Networks](http://arxiv.org/abs/1409.3215)
