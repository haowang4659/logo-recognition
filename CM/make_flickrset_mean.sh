#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

lmdb_root=/workspace/wanghao/git-repository/project/flickr_32/CM/lmdb
Outfile=/workspace/wanghao/git-repository/project/flickr_32/CM
TOOLS=/home/wanghao/caffe-sdk/caffe-master/build/tools
rm flickr_train_mean.binaryproto
rm flickr_test_mean.binaryproto
#rm flickr_sample_mean.binaryproto
$TOOLS/compute_image_mean $lmdb_root/img_train_lmdb \
  $Outfile/flickr_train_mean.binaryproto

$TOOLS/compute_image_mean $lmdb_root/img_test_lmdb \
  $Outfile/flickr_test_mean.binaryproto

#$TOOLS/compute_image_mean $lmdb_root/img_spl_lmdb \
#  $Outfile/flickr_spl_mean.binaryproto


echo "Done."
