#!/usr/bin/sh
Image=/workspace/wanghao/git-repository/original_data/FlickrLogos_crop
Outfile=/workspace/wanghao/git-repository/project/flickr_32/CM/lmdb
rm -rf $Outfile/img_train_lmdb
rm -rf $Outfile/img_test_lmdb
rm -rf $Outfile/img_sample_lmdb
caffe_root=/workspace/wanghao/git-repository/caffe-master
$caffe_root/build/tools/convert_imageset --shuffle \
--resize_width=256 \
--resize_height=256 \
$Image/classes/ $Image/train.txt $Outfile/img_train_lmdb
$caffe_root/build/tools/convert_imageset --shuffle \
--resize_width=256 \
--resize_height=256 \
$Image/classes/ $Image/test.txt $Outfile/img_test_lmdb

$caffe_root/build/tools/convert_imageset --shuffle \
--resize_width=256 \
--resize_height=256 \
$Image/sample/ $Image/sample.txt $Outfile/img_sample_lmdb

