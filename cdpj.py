from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from PIL import Image, ImageEnhance
import sys,os
from unet import Unet
import datetime
import time
from nets.makedir import mkdir
from nets.deldir import deldir
from nets.tpfg import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

# 构造参数解析器并解析参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images",default="img/", type=str, #required=True,
                help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", default="imgout/",type=str, #required=True,
                help="path to the output image")
ap.add_argument("-c", "--cache", default="VOCdevkit/VOC2007/",type=str, #required=True,
                help="path to the cache")
args = vars(ap.parse_args())  # vars函数是实现返回对象object的属性和属性值的字典对象

print(args)  # {python cdpj.py --images E:/datasetroot/img/ --output E:/datasetroot/outimg/ --cache E:/datasetroot/}
# 匹配输入图像的路径并初始化我们的图像列表
# rectangular_region = 2
Image.MAX_IMAGE_PIXELS = 2300000000




print("[INFO] loading images...")
if __name__ == "__main__":
    mkdir(args['images'])
    mkdir(args['output'])
    mkdir(args['cache'] + '/out/')
    mkdir(args['cache'] + '/out1/')
    mkdir(args['cache'] + '/imgout/')
    #deeplab = DeeplabV3()
    unet = Unet()
    count = False
    name_classes = ["background","root"]
# 获取到每张待拼接图像并排序，如['第一张图片路径'， 第二张图片路径'，第三张图片路径']
    x = os.listdir(args["images"])
# print(imagePaths)
# imagePaths = ['IMG_1786-2.jpg',
# 			  'IMG_1787-2.jpg',
# 			  'IMG_1788-2.jpg']
    num = 0
    j = 0

# 遍历图像路径，加载每个路径，然后将它们添加到我们的路径中图像到stich列表
    for j in x:
        t1 = datetime.datetime.now().microsecond
        t2 = time.mktime(datetime.datetime.now().timetuple())
    #     image = Image.open(args["images"] + x[num])
    #     # 图像整体预测
    #     f_image = deeplab.detect_image(image, count=count, name_classes=name_classes)
    #     f_image.save(args['cache'] + '/imgout/' + x[num])
    #     f_image = cv2.imread(args['cache'] + '/imgout/' + x[num])
        image_name = x[num].strip('.jpg')
    #     clip(f_image, 512, image_name, args['cache'] + '/imgout1/')

        # 图像分割成512预测
        
        image = cv2.imread(args['images'] + x[num])
        
        h_count, v_count = clip(image, 512, image_name, args['cache'] + '/out/')
        f = os.listdir(args['cache'] + '/out/')
        n = 0
        i = 0
        kernel = np.ones((6, 6), np.uint8)
        for i in f:
            a = Image.open(args['cache'] + '/out/' + f[n])   
            # b = cv2.imread(args['cache'] + '/imgout1/' + f[n])  
            # if np.all(b == 0):
            #     r_image = Image.new('RGB',(512,512),(0,0,0)) 
            #     r_image.save(args['cache'] + '/out1/' + f[n])
            # else:
                
            r_image = unet.detect_image(a, count=count, name_classes=name_classes)
            #     extrema = r_image.convert("L").getextrema()
            #     if extrema == (0,0):
            #         b = cv2.erode(b, kernel, iterations=1)
            #         cv2.imwrite(args['cache'] + '/out1/' + f[n] , b)
            #     else:
            # #r_image.show()
            r_image.save(args['cache'] + '/out1/' + f[n])
            n = n + 1
        IMAGES_PATH = args['cache'] + '/out1/' # 图片集地址
        IMAGES_FORMAT = ['.jpg', '.jpg'] # 图片格式
        height, weight = image.shape[:2]
        IMAGE_SIZE = 512
       
            
        IMAGE_ROW = h_count # 图片间隔，也就是合并成一张图后，一共有几行
        IMAGE_COLUMN = v_count # 图片间隔，也就是合并成一张图后，一共有几列
        IMAGE_SAVE_PATH = args['output'] + x[num] # 图片转换后的地址
        to_image = Image.new('RGB', (IMAGE_ROW * IMAGE_SIZE, IMAGE_COLUMN * IMAGE_SIZE)) #创建一个新图
        print(to_image.size)
        print(IMAGE_ROW,IMAGE_COLUMN)
        indey = 0
 
        for j in range(0, IMAGE_COLUMN):
            for i in range(0, IMAGE_ROW):
                from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                to_image.paste(from_image, (i*IMAGE_SIZE, j*IMAGE_SIZE))
                indey = indey + 1
        to_image = to_image.crop((0,0,weight,height))
        to_image.save(IMAGE_SAVE_PATH) # 保存新图

     
        # a = cv2.imread(args['cache'] + '/imgout/' + x[num])

        # b = cv2.imread(args['cache'] + '/out2/' + x[num])
        #kernel = np.ones((5,5),np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(9,9),anchor=None)
        # bp = cv2.dilate(b,kernel,iterations=1)
        # ap = cv2.erode(a,kernel,iterations=1)


        # im=cv2.add(a,b)
        
        # cv2.imwrite(args['output']+x[num], im)
        print('处理完成：' + str(x[num]))
        t3 = datetime.datetime.now().microsecond
        t4 = time.mktime(datetime.datetime.now().timetuple())
        strTime = 'funtion time use:%dms' % ((t4 - t2) * 1000 + (t3 - t1) / 1000)
        print(strTime)
    
    
    
        num = num + 1
    deldir(args['cache'] + '/out/')
    deldir(args['cache'] + '/out1/')
    # deldir(args['cache'] + '/out2/')
    #deldir(args['cache'] + '/imgout/')
    print('删除缓存成功')
    
    