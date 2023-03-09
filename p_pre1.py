import time

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import sys,os
from unet import Unet
import datetime
import time


Image.MAX_IMAGE_PIXELS = 2300000000
# cut_num = 4 # 4*4=16个图片
# #将图片填充为正方形
# def fill_image(image):
#     width, height = image.size  
#     #选取长和宽中较大值作为新图片的
#     new_image_length = width if width > height else height  
#     #生成新图片[白底]
#     #new_image = Image.new(image.mode, (new_image_length, new_image_length), color='white')  
#     new_image = Image.new(image.mode, (new_image_length, new_image_length))
#     #将之前的图粘贴在新图上，居中 
#     if width > height:#原图宽大于高，则填充图片的竖直维度
#     #(x,y)二元组表示粘贴上图相对下图的起始位置
#       new_image.paste(image, (0, int((new_image_length - height) / 2)))
#     else:
#       new_image.paste(image, (int((new_image_length - width) / 2),0))  
#     return new_image 
# #切图
# def cut_image(image):
#     width, height = image.size
#     item_width = int(width / cut_num)
#     box_list = []  
#     # (left, upper, right, lower) 
#     for i in range(0,cut_num):#两重循环，生成图片基于原图的位置 
#       for j in range(0,cut_num):      
#         #print((i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width))
#         box = (j*item_width,i*item_width,(j+1)*item_width,(i+1)*item_width)
#         box_list.append(box)

#     image_list = [image.crop(box) for box in box_list]  
#     return image_list
# #保存
# def save_images(image_list):
#     index = 0 
#     for image in image_list:
#       image.save('E:/github/deeplabv3-plus-pytorch-main/VOCdevkit/VOC2007/out/'+ str(index) + '.jpg')
#       index = index + 1

        

 


if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到__init__函数里修改self.colors即可
    #-------------------------------------------------------------------------#
    unet = Unet()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #-------------------------------------------------------------------------#
    #   count               指定了是否进行目标的像素点计数（即面积）与比例计算
    #   name_classes        区分的种类，和json_to_dataset里面的一样，用于打印种类和数量
    #
    #   count、name_classes仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    count           = False
    # name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # name_classes    = ["background","cat","dog"]
    name_classes    = ["background","root"]
    #path = 'E:/github/deeplabv3-plus-pytorch-main/VOCdevkit/VOC2007/out/'
    outpath = 'imgout1/'
    inputpath = 'ceshi/'

    #img = input('Input image filename:')
    # x = os.listdir(inputpath)
    
    # j = 0
    # for j in x:
      


      #image = Image.open(inputpath + x[num])


 
      # image = fill_image(image)
      
      # image_list = cut_image(image)
      # save_images(image_list)


    
    f = os.listdir(inputpath)
    n = 0
    i = 0
    for i in f:
        t1 = datetime.datetime.now().microsecond
        t2 = time.mktime(datetime.datetime.now().timetuple())
                

        a = Image.open(inputpath + f[n])       
        r_image = unet.detect_image(a, count=count, name_classes=name_classes)
        #r_image.show()
        r_image.save(outpath + f[n])
        print('处理完成：' + str(f[n]))
        t3 = datetime.datetime.now().microsecond
        t4 = time.mktime(datetime.datetime.now().timetuple())
        strTime = 'funtion time use:%dms' % ((t4 - t2) * 1000 + (t3 - t1) / 1000)
        print(strTime)

        n = n + 1
      # IMAGES_PATH = 'E:/github/deeplabv3-plus-pytorch-main/VOCdevkit/VOC2007/out1/' # 图片集地址
      # IMAGES_FORMAT = ['.jpg', '.jpg'] # 图片格式
      # width, height = image.size
      # new_image_length = width if width > height else height 
      # item_width = int(width / cut_num)
      # IMAGE_SIZE = item_width
      # IMAGE_ROW = 4 # 图片间隔，也就是合并成一张图后，一共有几行
      # IMAGE_COLUMN = 4 # 图片间隔，也就是合并成一张图后，一共有几列
      # IMAGE_SAVE_PATH = 'E:/github/deeplabv3-plus-pytorch-main/VOCdevkit/VOC2007/out2/' + x[num] # 图片转换后的地址
      # to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE)) #创建一个新图
      # indey = 0
 
      # for ii in range(0,IMAGE_ROW*IMAGE_COLUMN):
      #   if indey < 4:
      #     from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
      #     to_image.paste(from_image, (indey*IMAGE_SIZE, 0*IMAGE_SIZE))
      #   elif (indey >=4) & (indey < 8):
      #     from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
      #     to_image.paste(from_image, ((indey-4)*IMAGE_SIZE, 1*IMAGE_SIZE))
      #   elif (indey >=8) & (indey < 12):
      #     from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
      #     to_image.paste(from_image, ((indey-8)*IMAGE_SIZE, 2*IMAGE_SIZE))
      #   else:
      #     from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
      #     to_image.paste(from_image, ((indey-12)*IMAGE_SIZE, 3*IMAGE_SIZE))
      #   indey = indey +1
        # elif (indey >=48) & (indey < 60):
        #   from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
        #   to_image.paste(from_image, ((indey-48)*IMAGE_SIZE, 4*IMAGE_SIZE))
        # elif (indey >=60) & (indey < 72):
        #   from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
        #   to_image.paste(from_image, ((indey-60)*IMAGE_SIZE, 5*IMAGE_SIZE))
        # elif (indey >=72) & (indey < 84):
        #   from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
        #   to_image.paste(from_image, ((indey-72)*IMAGE_SIZE, 6*IMAGE_SIZE))
        # elif (indey >=84) & (indey < 96):
        #   from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
        #   to_image.paste(from_image, ((indey-84)*IMAGE_SIZE, 7*IMAGE_SIZE))
        # elif (indey >=96) & (indey < 108):
        #   from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
        #   to_image.paste(from_image, ((indey-96)*IMAGE_SIZE, 8*IMAGE_SIZE))
        # elif (indey >=108) & (indey < 120):
        #   from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
        #   to_image.paste(from_image, ((indey-108)*IMAGE_SIZE, 9*IMAGE_SIZE))
        # elif (indey >=120) & (indey < 132):
        #   from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
        #   to_image.paste(from_image, ((indey-120)*IMAGE_SIZE, 10*IMAGE_SIZE))
        # else:
        #   from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
        #   to_image.paste(from_image, ((indey-132)*IMAGE_SIZE, 11*IMAGE_SIZE))
        # indey = indey + 1
        
      
  

      # to_image.save(IMAGE_SAVE_PATH) # 保存新图
      # print('处理完成：' + str(x[num]))
      # num = num + 1    

