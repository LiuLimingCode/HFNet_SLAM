# HF-Net Model

This project is based on https://github.com/ethz-asl/hfnet.

The original HF-Net project was built with TensorFlow 1. However, Google Colab removed support for TensorFlow 1. The C++ and Python versions of TensorFlow 1 are no longer updated and are different to install. Therefore, to increase the compatibility of the HF-Net model, extra operations are needed.

## Convert model for yourself

1. Install TensorFlow 1 Python

You must install TensorFlow<=1.15. Here is an easy way to install the correct version.

```
pip install --upgrade pip
pip install nvidia-pyindex
pip install nvidia-tensorflow[horovod]
pip install nvidia-tensorboard==1.15
```

2. Download the checkpoint files from [here](https://projects.asl.ethz.ch/datasets/doku.php?id=cvpr2019hfnet).

3. Export the model

```
python3 export_model.py path_to_checkpoint_dir path_to_output_dir
```

## What's the differences?

1. The original HF-Net needs the support of tf.contrib.resampler, which is no longer supported. Therefore, the resampler model is removed and re-implemented in the BasedModel.cc file.

2. To improve the efficiency, the number of NMS iterations is reduced to 2 from 3.

