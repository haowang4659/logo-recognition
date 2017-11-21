#!/usr/bin/env sh

CAFFE_TOOLS=/home/wanghao/caffe-sdk/old-caffe-master/build/tools
SOLVER_ROOT=/workspace/wanghao/git-repository/project/flickr_32/CM
$CAFFE_TOOLS/caffe train -solver $SOLVER_ROOT/solver.prototxt -weights $SOLVER_ROOT/VGG_CNN_M_1024.v2.caffemodel
