import tensorflow as tf
import cv2
import os
from matplotlib import pyplot as plt


#prevent tensorflow from allocating all gpu memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    #for each gpu, dont allocate all memory, limit
    tf.config.experimental.set_memory_growth(gpu, True)


data_dir = 'data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                print('Image not loaded properly {}'.format(image_path))
                os.remove(image_path)
                print('Removed {}'.format(image_path))
                continue
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            os.remove(image_path)
            print('Removed {}'.format(image_path))




data = tf.keras.utils.image_dataset_from_directory('data')

