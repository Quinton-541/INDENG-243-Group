import pandas as pd
import numpy as np
import os
import keras.api._v2.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

import cv2

if __name__ == "__main__":

    image_path='Data/img_align_celeba/img_align_celeba'
    image_files=os.listdir(image_path)
    train_images=[]
    for file in image_files: #读取所有图片文件
        img_path = os.path.join(image_path, file)
        img = cv2.imread(img_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将 BGR 转换为 RGB
        train_images.append(img)
    test_images=np.array(train_images)[10001:13000]
    train_images=np.array(train_images)[:10000]

    landmark = pd.read_csv('Data/list_landmarks_align_celeba.txt', sep='\s+', index_col=0) #读取左右眼标签信息
    train_labels=[]
    for image in landmark.index:
        temp=landmark.loc[image,['lefteye_x','lefteye_y','righteye_x','righteye_y']]
        train_labels.append(temp.tolist())
    test_labels=np.array(train_labels)[10001:13000]
    train_labels = np.array(train_labels)[:10000]

    print('Data_load_Complete')
    #
    # # 定义模型
    # model = Sequential([
    #     # 卷积层
    #     Conv2D(32, (3, 3), activation='relu', input_shape=(218, 178, 3)),
    #     MaxPooling2D(2, 2),
    #     Conv2D(64, (3, 3), activation='relu'),
    #     MaxPooling2D(2, 2),
    #     Conv2D(128, (3, 3), activation='relu'),
    #     MaxPooling2D(2, 2),
    #     # 展平层
    #     Flatten(),
    #     # 全连接层
    #     Dense(128, activation='relu'),
    #     Dropout(0.5),
    #     # 输出层，4个单元对应于两个眼睛的x和y坐标
    #     Dense(4)
    # ])
    #
    # model.compile(optimizer='adam',
    #               loss='mean_squared_error',
    #               metrics=['accuracy'])
    #
    # print("Learning_process_start")
    # model.fit(train_images, train_labels, epochs=20, verbose=1)
    #
    # test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    #
    # print(test_loss,test_acc)
    #
    # model.save('Data/eye_pred', save_format='tf')

    model=keras.models.load_model('Data/eye_pred')
    test_res=model.predict(test_images)
    print(test_labels)
    print(test_res)

    for i in range(0,3000):
        target_image=test_images[i]
        actual_label=test_labels[i]
        pred_label=[round(num) for num in test_res[i]]

        center_l_x,center_l_y,center_r_x,center_r_y=actual_label[0],actual_label[1],actual_label[2],actual_label[3] #实际位置

        left_l,right_l,top_l,bottom_l=center_l_x-5,center_l_x+5,center_l_y-5,center_l_y+5 #左眼
        cv2.rectangle(target_image, (left_l, top_l), (right_l, bottom_l), (255, 0, 0), 1)  # 红色方框，宽度为1个像素
        left_l,right_l,top_l,bottom_l=center_r_x-5,center_r_x+5,center_r_y-5,center_r_y+5 #右眼
        cv2.rectangle(target_image, (left_l, top_l), (right_l, bottom_l), (255, 0, 0), 1)

        center_l_x, center_l_y, center_r_x, center_r_y = pred_label[0], pred_label[1], pred_label[2], pred_label[3] #预测位置

        left_l, right_l, top_l, bottom_l = center_l_x - 5, center_l_x + 5, center_l_y - 5, center_l_y + 5
        cv2.rectangle(target_image, (left_l, top_l), (right_l, bottom_l), (0, 255, 0), 1)  # 绿色方框，宽度为1个像素
        left_l, right_l, top_l, bottom_l = center_r_x - 5, center_r_x + 5, center_r_y - 5, center_r_y + 5  # 右眼
        cv2.rectangle(target_image, (left_l, top_l), (right_l, bottom_l), (0, 255, 0), 1)

        # 显示或保存图片
        cv2.imshow('Image with Rectangle', target_image)
        cv2.waitKey(0)  # 等待按键后关闭窗口





