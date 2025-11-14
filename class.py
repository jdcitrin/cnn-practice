import tensorflow as tf
import cv2
import os
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import numpy as np

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
class_names = data.class_names  # save class names before mapping
print(f"Class names: {class_names}")
data = data.map(lambda x,y: (x/255, y))

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

def augment(image, label):
    return data_augmentation(image, training=True), label


#1 sad
#0 happy
"""
fig, anexts = plt.subplots(ncols=5, figsize=(10,10))
for idx, img in enumerate(batch[0][:5]):
   anexts[idx].imshow(img) #cant be int
   anexts[idx].title.set_text(batch[1][idx])
plt.show()
"""

train_size = int(len(data)*.7) 
val_size = int(len(data)*.2)+1
test_size = int(len(data)*.1)+1

'''
print(len(data))
print(train_size)
print(val_size)
print(test_size)
'''


#allocating data
train = data.take(train_size)
train = train.map(augment)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


"""
print("\n=== Checking Labels ===")
sample_batch = train.take(1).as_numpy_iterator().next()
print(f"Class names: {class_names}")
print(f"Sample labels in batch: {sample_batch[1]}")

# Visualize to confirm labels match images
fig, axes = plt.subplots(ncols=5, figsize=(15,5))
for idx in range(5):
    axes[idx].imshow(sample_batch[0][idx])
    axes[idx].set_title(f"Label: {sample_batch[1][idx]} ({class_names[sample_batch[1][idx]]})")
plt.show()
"""

#data preprocessed for deep learning model using keras sequential
model_path = 'models/happy_class.h5'

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:

    model = Sequential()
    model.add(Conv2D(32, (3,3), 1, activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.5)) 
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile('adam', loss = tf.losses.BinaryCrossentropy(), metrics = ['accuracy'])
    #print(model.summary())


    ldir = 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = ldir)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    #1 epoch is 1 run over trainingdata
    #validation data = evaluation
    hist = model.fit(train, epochs=50, validation_data = val, callbacks=[tensorboard_callback, early_stopping])
    os.makedirs('models', exist_ok=True)
    model.save('models/happy_class.h5')

#print(hist.history)

"""
fig = plt.figure()
plt.plot(hist.history['loss'], color = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], color = 'blue', label = 'val_loss')
fig.suptitle('Loss', fontsize= 20)
plt.legend(loc = "upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color = 'red', label = 'acc')
plt.plot(hist.history['val_accuracy'], color = 'blue', label = 'val-a')
fig.suptitle('acc', fontsize= 20)
plt.legend(loc = "upper left")
plt.show()
"""

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X,y = batch
    yhat = model.predict(X)
    pre.update_state(y,yhat)
    re.update_state(y,yhat)
    acc.update_state(y,yhat)

    print(f'Precision: {pre.result().numpy():.4f}')
    print(f'Recall: {re.result().numpy():.4f}')
    print(f'Accuracy: {acc.result().numpy():.4f}')

img = cv2.imread('./hap_test/(10).png')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
resize = tf.image.resize(img,(256,256))
plt.imshow(resize.numpy().astype(int))

yhat = model.predict(np.expand_dims(resize/255, 0))
print(yhat)

#from tensorflow.keras.models import load_models
