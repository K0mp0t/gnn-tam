# Graph Neural Networks with Trainable Adjacency Matrices for Fault Diagnosis on Multivariate Sensor Data

This repository is the official implementation of model architectures from the paper [Graph Neural Networks with Trainable Adjacency Matrices for Fault Diagnosis on Multivariate Sensor Data](https://doi.org/10.1109/ACCESS.2024.3481331).

## Training and inference examples

[FDDBenchmark](https://github.com/AIRI-Institute/fddbenchmark) was used in our experiments.

Training step:

```
python train.py
```

Inference step:

```
python evaluate.py
```

## Citation

Please cite our paper as follows:

```
@article{kovalenko2024graph,
  title={Graph neural networks with trainable adjacency matrices for fault diagnosis on multivariate sensor data},
  author={Kovalenko, Aleksandr and Pozdnyakov, Vitaliy and Makarov, Ilya},
  journal={IEEE Access},
  year={2024},
  volume={12},
  pages={152860-152872},
  publisher={IEEE}
}
```
