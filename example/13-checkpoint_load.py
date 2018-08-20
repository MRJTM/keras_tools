from keras.models import load_model
from keras.optimizers import SGD
import numpy as np
import cv2

"""---------------------------导入模型-------------------------------"""
model=load_model('../saved_models/weights-026-0.5365.hdf5')
# 编译一下模型
# opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
# model.compile(loss="categorical_crossentropy", optimizer=opt,
#               metrics=["accuracy"])

"""-----------------------------导入测试图片-------------------------------"""
img=cv2.imread('../test_images/example_03.jpg')
cv2.imshow('test_image',img)
# img=img[100:475,:,:]
# img=img[:,100:436,:]
cv2.imshow('crop_image',img)
img_resize=cv2.resize(img,(32,32))
img_input=np.reshape(img_resize,(1,32,32,3))
cv2.waitKey(0)

"""------------------------------输入网络进行测试-------------------------------------"""
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
	"dog", "frog", "horse", "ship", "truck"]
predict=model.predict(img_input)
print("predict:",predict)
result=labelNames[np.argmax(predict)]
print("result:",result)


