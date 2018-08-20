from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np

image_path='../test_images/horse.jpg'
output_path='../output/augmentation_demo'

# 导入图片
print("[INFO] loading example image...")
image = load_img(image_path)
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# 实例化数据增强器
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                        horizontal_flip=True, fill_mode="nearest")

# 不断输出增强的图片
print("[INFO] generating images...")
imageGen = aug.flow(image, batch_size=1, save_to_dir=output_path,
                    save_prefix='aug', save_format="jpg")

total=0
for image in imageGen:
    total+=1
    if total==10:
        break