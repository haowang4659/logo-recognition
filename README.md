# logo-recognition
one-shot learning logo recognition <br /><br /><br />
运行环境说明：<br />
  运行环境是在ubuntu14.0.4以及配置了caffe的gpu版本<br /><br />
文件夹说明：<br />
CM：该文件夹下存储的是第一阶段中训练VGG_M的代码<br />
  CM/create_lmdb.sh ：该文件是制作数据集的文件<br />
  CM/extract_feature.py ：该文件是训练完网络后提取特征的代码<br />
  CM/make_flickrset_mean.sh ：该文件是制作数据集的均值的文件<br />
  CM/flickrset_mean.binaryproto ：该文件是数据集的均值，由make_flickrset_mean.sh脚本生成<br />
  CM/train_val.prototxt ：该文件是VGG_M的网络结构文件<br />
  CM/solver.prototxt ：该文件是训练文件<br /><br />
CN：该文件夹下存储的是第二阶段中训练比对网络的代码<br />
  CN/EF.zip ：该压缩包是第一阶段提取的特征，由extract_feature.py文件生成<br />
  CN/extract_feature.py ：该文件是训练完网络后提取特征的代码<br />
  CN/train_val.prototxt ：该文件是比对网络的网络结构文件<br />
  CN/solver.prototxt ：该文件是训练文件<br />
  CN/run.sh ：该文件是运行脚本<br /><br />
CW：该文件夹下存储的是加权的余弦距离的比对网络的代码<br />
  CW/EF.zip ：该压缩包是第一阶段提取的特征，由extract_feature.py文件生成<br />
  CW/EW.zip ：该压缩包是第一阶段提取的权值，由extract_weight.py文件生成<br />
  CW/extract_feature.py ：该文件是训练完网络后提取特征的代码<br />
  CW/extract_weight.py ：该文件是训练完网络后提取全连接层权值的代码<br />
  CW/train_val.prototxt ：该文件是加权比对网络的网络结构文件<br />
  CW/solver.prototxt ：该文件是训练文件<br />
  CW/run.sh ：该文件是运行脚本<br />
