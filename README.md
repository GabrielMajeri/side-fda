# PyTorch implementation of "Single-Image Depth Estimation Using Fourier Domain Analysis"

## Description

This repository contains an open source implementation of a CVPR 2018 paper
which reconstructs the depth of a scene based on a color image of it.

The paper achieves competitive results by:
- reusing an existing CNN architecture (ResNet-152) and extending it with some new,
  specialized convolutional layers

- introducing a new loss function which improves the accuracy of the depth estimation

- running the network on the image and on smaller cropped sections of the image, then
  recombining the results using Fourier domain analysis

[cvpr]: http://openaccess.thecvf.com/content_cvpr_2018/papers/Lee_Single-Image_Depth_Estimation_CVPR_2018_paper.pdf

## Implementation details

The original implementation, [available on the authors' site][homepage],
was written in MATLAB and used Caffe.
This implementation is based on the PyTorch library for doing deep learning in Python.

[homepage]: http://mcl.korea.ac.kr/~jaehanlee/depth/index.html

## License

The code in this repository is licensed under the Mozilla Public License Version 2.0,
see the [LICENSE](LICENSE.txt) file for the full text.
