# AITISA工作组PyTorch平台基准代码(检测部分)

## 简介
本仓库为增量正则化剪枝算法在检测任务(pytorch—faster-rcnn)上的代码实现。

## 环境
- Python == 3.6.7
- PyTorch == 0.4.0
- Torchvision == 0.2.1
- CUDA == 9.0
- scipy == 1.2.1
- pillow == 6.1.0

## 配置与安装

### Data Preparation
* **PASCAL_VOC 07+12**: First of all, download the dataset in [pascal_voc](https://pjreddie.com/projects/pascal-voc-dataset-mirror/).
And follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets.

### Compilation

As pointed out by [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), choose the right `-arch` in `make.sh` file, to compile the cuda code:

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |
 
### Install
Install all the python dependencies using pip:
```
pip install -r requirements.txt
```
Compile the cuda dependencies using following simple commands:
```
cd lib && sh make.sh
```
It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop.

## 参数
- `--net`：       模型(default:vgg16)
- `--dataset`：   数据集(pascal_voc 2007)
- `--bs`：        训练集batch_size大小(default: 1)
- `--lr`：        学习率(default: 0.001)
- `--sparse_reg`：稀疏正则化剪枝算法(采用增量正则化方法)
- `--rate`：      各层剪枝率
- `--skip_idx`：  设定不剪枝层
- `--state`：     剪枝流程初始状态(default: 'prune')
- `--save_dir`:   输出模型及log日志的保存目录

## 剪枝及重训练

### 示例（压缩率=50%）
```
nohup python -u prune_trainval.py --net vgg16 --dataset pascal_voc --bs 1 --lr 0.001 --sparse_reg True --rate 0.5 --skip True --load_path ${path to pretrained model}--save_dir ${dir to save weights} > prune_weights/vgg16_2x_prune_output.log 2>&1 &

```

## log日志查看

### 查看当前剪枝率
```
cat prune_weights/vgg16_2x_prune_output.log | grep "prune"
```
### 查看log日志末尾10行
```
tail prune_weights/vgg16_2x_prune_output.log -n 10
```

## 测试准确率
```
python test.py --net vgg16 --dataset pascal_voc --load_path ${dir to save weights}/faster_rcnn_1_10_10021.pth
```

## 检测剪枝率
```
python check_prune.py --net vgg16 --dataset pascal_voc --weights checkpoints/vgg16_2.0x_epoch10_mAP0.6811.pth 
```

## 模型精度

### VGG16
network | prune rate |  mAP  | speedup | download_url | Extraction code
--------|------------|-------|---------|--------------|------------------
baseline | 0.00 | 0.7011 | 1.00x | https://pan.baidu.com/s/1fRJhtt-k64H1RAZ-R_uFhw | mza2
compression| 0.50 | 0.6811 | 2.00x | https://pan.baidu.com/s/1en3Wl-lMuLR-IYdAdIwjRQ | u5fy
