# AcfNet
This repository contains the code (in PyTorch) for "[Adaptive Unimodal Cost Volume Filtering for Deep Stereo Matching](https://arxiv.org/abs/1909.03751)", accepted in AAAI 2020.

## Notes

* We have provided the modules used in our AcfNet.
* Specially, a DenseMatchingBenchmark is coming, mainly designed for stereo matching task.
* The architecture is based on two wellknown detection framework: [mmdetection](https://github.com/open-mmlab/mmdetection) and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). By integrating their major features, our architecture is suitable for dense matching(e.g. stereo matching), and achieves robust performance！


## Requirements:
- PyTorch1.1+, Python3.5+, Cuda10.0+

## Reference:

If you find the code useful, please cite our paper:

    @article{zhang2019adaptive,
      title={Adaptive Unimodal Cost Volume Filtering for Deep Stereo Matching},
      author={Zhang, Youmin and Chen, Yimin and Bai, Xiao and Zhou, Jun and Yu, Kun and Li, Zhiwei and Yang, Kuiyuan},
      journal={arXiv preprint arXiv:1909.03751},
      year={2019}
    }
