from keras.models import load_model
from keras.models import Model
import cv2
import numpy as np

# 配置相关路径
image_path='../test_images/hashiqi.jpg'
model_path='../saved_models/animals_fine_tune1.model'

# 载入模型
model=load_model(model_path)

# 载入图片并进行预处理
img=cv2.imread(image_path)
image=cv2.resize(img,(224,224))
image=np.reshape(image,(1,224,224,3))

# 用网络进行预测
label_names=['cat','dog','panda']
predict_result=model.predict(image)

# 输出预测结果
predict_label=label_names[np.argmax(predict_result)]
print("predicted result:",predict_label)
prob=np.max(predict_result)

# 将结果显示在测试图片上
cv2.putText(img,"{}:{}".format(predict_label,prob),(20,40),
            cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,
            color=(0,0,255),thickness=1)

cv2.imshow("result",img)
cv2.waitKey(0)