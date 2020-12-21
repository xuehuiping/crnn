# -*- coding: utf-8 -*-
# author: huihui
# date: 2020/12/20 8:45 下午 

import argparse
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import models.crnn as crnn


# 类别对应的字母表，如 0123456789abcdefghijklmnopqrstuvwxyz
alphabet = open('alphabet2.txt').read()

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='./expr/test2/CRNN_38_8000.pth', help='训练好的模型参数，如./data/crnn.pth')
parser.add_argument('--img_path', default='./data/test/word_287.png', help='需要测试的图片，如./data/demo.png')
opt = parser.parse_args()

# imgH(图片高度), nc(图片通道数), nclass(类别数), nh(隐层数)
model = crnn.CRNN(
    imgH=32,
    nc=1,
    nclass=len(alphabet) + 1,
    nh=128
)

model_path = opt.model_path
img_path = opt.img_path

print('loading pretrained model from %s' % model_path)
if torch.cuda.is_available():
    model = model.cuda()
    model.load_state_dict(torch.load(model_path))
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)

converter = utils.strLabelConverter(alphabet)

# 图片处理
transformer = dataset.resizeNormalize((192, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

# 输入模型并预测结果
model.eval()
preds = model(image)
_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

# 对结果进行解码，类别数和字母表中字符对应，并输出结果
preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
