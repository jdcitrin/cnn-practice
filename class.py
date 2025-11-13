import tensorflow as tf
import cv2
import os
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


#prevent tensorflow from allocating all gpu memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    #for each gpu, dont allocate all memory, limit
    tf.config.experimental.set_memory_growth(gpu, True)


data_dir = 'data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']


#removing unfit images
'''
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
'''



#load dataset from directory, generates labels automatically
data = tf.keras.utils.image_dataset_from_directory('data', batch_size = 25)

data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()

#1 sad
#0 happy






scaled = batch[0]/255 # values betwwen 0-1
scaled.max()

data = data.map(lambda x,y: (x/255, y))
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()


#images represented as numpy arrays
##print(batch[0].shape) #images
##print(batch[1]) #labels
'''
fig, anexts = plt.subplots(ncols=5, figsize=(10,10))
for idx, img in enumerate(batch[0][:5]):
   anexts[idx].imshow(img) #cant be int
   anexts[idx].title.set_text(batch[1][idx])
plt.show()
'''
train_size = int(len(data)*.7) 
val_size = int(len(data)*.2)+1
test_size = int(len(data)*.1)+1

'''
print(len(data))
print(train_size)
print(val_size)
print(test_size)
'''

#25 in each batch
#setting partitions for training model. 


#allocating data
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


#data preprocessed for deep learning model using keras sequential

