import cv2
# ---------------------------------------------------#
#   裁剪：
#       img：图片（cv格式）
#       target_size: 目标尺寸(仅支持正方形)
#       file_name: 不含扩展名的文件名
#       pic_out_path: 输出文件夹
#       padding: 填充颜色(B,G,R)
# ---------------------------------------------------#
def clip(img, target_size, file_name, pic_out_path, padding=(0, 0, 0)):
    max_y, max_x = img.shape[0], img.shape[1]
    # 若不能等分，则填充至等分
    if max_x % target_size != 0:
        padding_x = target_size - (max_x % target_size)
        img = cv2.copyMakeBorder(img, 0, 0, 0, padding_x, cv2.BORDER_CONSTANT, value=padding)
        max_x = img.shape[1]
    if max_y % target_size != 0:
        padding_y = target_size - (max_y % target_size)
        img = cv2.copyMakeBorder(img, 0, padding_y, 0, 0, cv2.BORDER_CONSTANT, value=padding)
        max_y = img.shape[0]
 
    h_count = int(max_x / target_size)
    v_count = int(max_y / target_size)
 
    count = 0
    for v in range(v_count):
        for h in range(h_count):
            x_start = h * target_size
            x_end = (h + 1) * target_size
            y_start = v * target_size
            y_end = (v + 1) * target_size
            cropImg = img[y_start:y_end, x_start:x_end]  # 裁剪图像
            target_path = pic_out_path + str(count) + '.jpg'
            cv2.imwrite(target_path, cropImg)  # 写入图像路径
            count += 1
    return h_count, v_count