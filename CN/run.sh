#!/usr/bin/env sh

CAFFE_TOOLS=/workspace/wanghao/git-repository/caffe-CN/build/tools
SOLVER_ROOT=/workspace/wanghao/git-repository/project/flickr_32/CN
$CAFFE_TOOLS/caffe train -solver $SOLVER_ROOT/solver.prototxt -weights $SOLVER_ROOT/final.caffemodel
