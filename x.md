## 机器配置
```
Ubuntu  16.04 64位
NVIDIA V100
```

## 数据
```
xuehp@haomeiya007:~/git/crnn$ ls data
Challenge2_Training_Task3_Images_GT  ICDAR2013_REC.zip	ICPR-MTWI-REC.zip  recognition
```
其中`ICPR-MTWI-REC`解压后的`recognition`需要处理`gt.txt`文件

## 运行

`conda create --name crnn python=3.6`

```
xuehp@haomeiya007:~/git/crnn$ source ~/anaconda3/bin/activate 
(base) xuehp@haomeiya007:~/git/crnn$ conda activate crnn
```
```pip install torch torchvision numpy matplotlib```

```
source ~/anaconda3/bin/activate 
conda activate crnn
```

## 数据集1`ICDAR2013_REC.zip`

`Challenge2_Training_Task3_Images_G`

### 训练

请修改对应参数

```
如有GPU，添加参数cuda，否则不需要加
(crnn) xuehp@haomeiya007:~/git/crnn$ python train.py \
--trainroot ./data/Challenge2_Training_Task3_Images_GT/ \
--valroot ./data/Challenge2_Training_Task3_Images_GT/ \
--cuda \
--ngpu 1 \
--expr_dir expr1 >> log.train.1.12.21.txt &
```

### 测试

请修改对应参数

```python demo1.py \
--model_path ./expr/CRNN_499_50.pth \
--img_path ./data/test/word_287.png \
```

见`result/test1`

## 数据集2`ICPR-MTWI-REC.zip`

`xuehp@haomeiya007:/home/xuehp/git/crnn/data$ unzip ICPR-MTWI-REC.zip`

### 训练

请修改对应参数

```
如有GPU，添加参数cuda，否则不需要加
(crnn) xuehp@haomeiya007:~/git/crnn$ python train.py \
--trainroot ./data/recognition/train/ \
--valroot ./data/recognition/train/ \
--cuda \
--ngpu 1 \
--expr_dir expr2 >> log.train.2.12.21.txt &
```
```
(crnn) xuehp@haomeiya007:~/git/crnn$ python train.py \
--trainroot ./data/recognition/train/ \
--valroot ./data/recognition/train/ \
--batchSize 64 \
--saveInterval 5 \
--cuda \
--expr_dir expr2
```
### 测试

请修改对应参数

```python demo2.py \
--model_path ./expr/CRNN_50.pth \
--img_path ./data/test/word_287.png \
```

见`result/test2`
这个数据集比较大，如果使用CPU，训练时间将以月计。

