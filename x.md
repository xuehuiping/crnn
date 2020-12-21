
## 数据
```
xuehp@haomeiya007:~/git/crnn$ ls data
Challenge2_Training_Task3_Images_GT  ICDAR2013_REC.zip
```

## 运行

`conda create --name crnn python=3.6`

```
xuehp@haomeiya007:~/git/crnn$ source ~/anaconda3/bin/activate 
(base) xuehp@haomeiya007:~/git/crnn$ conda activate crnn
```

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
--cuda
```
### 测试

请修改对应参数

```python demo.py \
--model_path ./expr/CRNN_499_50.pth \
--img_path ./data/test/word_287.png \
```

见`result/test1`

## 数据集2`ICPR-MTWI-REC`

`xuehp@haomeiya007:/home/xuehp/git/crnn/data$ unzip ICPR-MTWI-REC.zip`

### 训练

请修改对应参数

```
如有GPU，添加参数cuda，否则不需要加
(crnn) xuehp@haomeiya007:~/git/crnn$ python train.py \
--trainroot ./data/recognition/train/ \
--valroot ./data/recognition/train/ \
--cuda
```
### 测试

请修改对应参数

```python demo.py \
--model_path ./expr/CRNN_499_50.pth \
--img_path ./data/test/word_287.png \
```

见`result/test1`
这个数据集比较大，如果使用CPU，训练时间将以月计。

