The code is created based on the method described in the following paper

[1] Zheng, Qingqing, Fengyuan Zhu, and Pheng-Ann Heng. "Robust Support Matrix Machine for Single Trial EEG Classification." IEEE Transactions on Neural Systems and Rehabilitation Engineering 26.3 (2018): 551-562.

The code and the algorithm are for non-comercial use only.

Author: Qingqing Zheng (qqzheng@cse.cuhk.edu.hk)

Date : 07/09/2018

Version : 1.0

Copyright 2018, The Chinese University of Hong Kong.

This folder contains the following files:

RSMM.pdf                 : The paper

libqp                    : library for quadratic programming

binary_rsmm_demo.m       : a demo for binary matrix classification

multi_rsmm_demo.m        : a demo for multiclass matrix classification

SparLR_ADMM.m            : the robust support matrix machine (Algorithm 1 in paper)

rpca.m                   : feature recovery in paper

shrinkage.m              : proximal operator for low rank

multi_prediction.m       : multiclass evaluation 


