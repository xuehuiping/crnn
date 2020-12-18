
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
### 训练

请修改对应参数

```
如有GPU，添加参数cuda，否则不需要加
(crnn) xuehp@haomeiya007:~/git/crnn$ python train.py --cuda
```
### 测试

请修改对应参数

`python demo.py`