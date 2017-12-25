# pip install face_recognition pillow

import face_recognition
from PIL import Image, ImageDraw
import math
import random


class Config:
    WithCPU = 'hog'
    WithCUDA = 'cnn'
    size_index = 2.5                        # 尺寸缩放系数
    hat_offset_index_h = (1 - 0.16)*0.75    # 垂直位置系数
    hat_offset_index_w = 165./450           # 水平位置系数
    rotate_index = 2                        # 旋转系数

# 计算若干点的平均值
def average(l):
    count = len(l)
    return (int(sum([x[0] for x in l])/count), int(sum([x[1] for x in l])/count))

# 计算大小
def get_size(info):
    x1, y1 = info[0]
    x2, y2 = info[1]
    x3, y3 = info[2]

    width = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    height = math.sqrt((x3 - x1)**2 + (y3 - y1)**2 - (width)**2/4) * Config.size_index
    width = height

    return (int(width), int(height))

# 计算旋转角度
def get_rot(info):
    x1, y1 = info[0]
    x2, y2 = info[1]

    x = int((math.atan((x1 - x2) / (y2 - y1)) * 180 / math.pi))
    x = x+90 if x < 0 else 90-x
    x = x*Config.rotate_index if math.fabs(x) < 20 else x

    return x

# 计算嘴唇的平均位置对于两眉毛平均位置形成的直线的对称点
def get_mirror_point(info):
    x1, y1 = info[0]
    x2, y2 = info[1]
    x3, y3 = info[2]

    x4 = (y3 + (x2 - x1)/(y2 - y1)*x3 - 2*y1 + y3 - (y2 - y1)/(x2 - x1)*(x3 - 2*x1)) / ((y2 - y1)/(x2 - x1) - (x1 - x2)/(y2 - y1))
    y4 = (x1 -x2)/(y2 - y1)*x4 + y3 + (x2 - x1)/(y2 - y1)*x3

    return (int(x4), int(y4))

# 计算贴图的坐标
def get_xy(info):
    x, y = get_mirror_point(info)
    w, h = get_size(info)

    return (int(x - w*Config.hat_offset_index_w), int(y - h*Config.hat_offset_index_h))


import sys
filename = sys.argv[1] 

image = face_recognition.load_image_file(filename)
img = Image.open(filename)
# 正常的帽子
img_h = Image.open('hat.png')
# 交换红色和绿色通道，得到一顶原谅色的帽子
r, g, b, a = img_h.split()
img_forgive = Image.merge("RGBA", (g, r, b, a))
hats = [img_h, img_forgive]

for face in face_recognition.face_landmarks(image):
    le = average(face['left_eyebrow'])
    re = average(face['right_eyebrow'])
    lipe = average(face['bottom_lip'])
    if re[1] == le[1]:
        # 眉毛的平均位置的Y坐标可能相等，这样会导致除零错误，所以这里相等则+1
        re = (re[0], re[1]+1)
    face_info = (le, re, lipe)

    # 随机选取一个帽子，缩放到合适大小
    this_img = random.choice(hats).resize(get_size(face_info), Image.ANTIALIAS)
    # 根据眉毛形成的直线的斜率，旋转帽子
    this_img = this_img.rotate(get_rot(face_info))
    # 把帽子贴到对应位置
    img.paste(this_img, get_xy(face_info), mask=this_img.split()[-1])
    
if len(sys.argv) == 3:
    img.save(sys.argv[2])
img.show()
