# lru_unofficial

LRU implementation in TensorFlow 2.0.
The LRU was introduced in [Resurrecting Recurrent Neural Networks for Long Sequences](https://dl.acm.org/doi/10.5555/3618408.3619518) at ICML, and 
belongs to the state-space models family, which are models able to handle extremely long sequences more
gracefully than attention based architectures. You can find [here](https://github.com/NicolasZucchet/minimal-LRU/blob/main/lru/model.py)
the JAX implementation that we took as a reference, as recommended by one of the authors.

JAX and PyTorch implementations to come. However, parallel scans are not implemented 
native in PyTorch, as noted [here](https://github.com/pytorch/pytorch/issues/95408).
However custom implementations exist, such as [this one](https://github.com/i404788/s5-pytorch/blob/74e2fdae00b915a62c914bf3615c0b8a4279eb84/s5/jax_compat.py).
