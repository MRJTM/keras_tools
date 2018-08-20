from keras.applications import VGG16
import numpy as np
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import cv2
import sys

"""-----------------------导入测试图片------------------------"""
image_path='test_images/example_03.jpg'
# image_path=sys.argv[1]
# img=cv2.imread(image_path)
# cv2.imshow("test iamge",img)
# img=cv2.resize(img,(224,224))
# img=img.astype(np.float64)
# img[:,:,0]-=np.mean(img[:,:,0])
# img[:,:,1]-=np.mean(img[:,:,1])
# img[:,:,2]-=np.mean(img[:,:,2])
# img=img.reshape((1,224,224,3))

image = load_img(image_path, target_size=(224, 224))
image = img_to_array(image)

# preprocess the image by (1) expanding the dimensions and
# (2) subtracting the mean RGB pixel intensity from the
# ImageNet dataset
image = np.expand_dims(image, axis=0)
image = imagenet_utils.preprocess_input(image)
image_input=[image]
image_input=np.vstack(image_input)



"""------------------------用VGG去抽取特征---------------------"""
model = VGG16(weights="imagenet", include_top=False)
feature_maps=model.predict(image_input,batch_size=1)
print("feature map size:",np.shape(feature_maps))

"""------------------------打印前100张特征图--------------------"""
plt.figure()
for i in range(10):
    for j in range(10):
        sub_image=feature_maps[:,:,:,i*10+j]
        sub_image=sub_image.reshape((7,7))
        # sub_image/=np.max(sub_image)
        plt.subplot(10,10,i*10+j+1)
        plt.imshow(sub_image)

plt.savefig('output/VGG_feature_map.jpg')
plt.show()

sub_image=feature_maps[:,:,:,1]
sub_image=sub_image.reshape((7,7))
sub_image=cv2.resize(sub_image,(28,28))
cv2.imshow("feature map:",sub_image)

cv2.waitKey(0)