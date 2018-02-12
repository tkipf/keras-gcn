Deep Learning on Graphs with Keras
====

Keras-based implementation of graph convolutional networks for semi-supervised classification.

Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907) (ICLR 2017)

For a high-level explanation, have a look at our blog post:

Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016)

**NOTE: This code is not intended to reproduce the experiments from the paper as the initialization scheme, dropout scheme, and dataset splits differ from the original implementation in TensorFlow: https://github.com/tkipf/gcn**

Installation
------------

```python setup.py install```

Dependencies
-----

  * keras (1.0.9 or higher)
  * TensorFlow or Theano

Usage
-----

```python train.py```

Dataset reference (Cora)
----------

[Sen et al., Collective Classification in Network Data, AI Magazine 2008](http://linqs.cs.umd.edu/projects/projects/lbc/)


## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N. and Welling, Max},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}
```
