import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) >0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

## Data Preprocessing

# import necessary libraries
import numpy as np
import glob, os, cv2
import matplotlib.pyplot as plt
import pyprind
import pandas as pd
import random
import pickle

img_dir ='C:/Users/Changwon Yoon/Desktop/Deep Learning/Project/airbus-ship-detection/train_v2/'
img_paths = glob.glob(os.path.join(img_dir,'*.jpg'))
img_paths[:5]

seg = pd.read_csv("train_ship_segmentations_v2.csv") # import encoded masks
seg = seg.fillna(0)
ship_df = pd.DataFrame(columns =['ImageId'])
ship_list = []
for i in range(len(img_paths)):
    ship_list.append(img_paths[i].split('\\')[-1])
ship_df['ImageId'] = ship_list

# Counting Ships
has_mask_df = seg.loc[seg['EncodedPixels'] !=0]
count_df = pd.DataFrame(has_mask_df['ImageId'].value_counts())
count_df.reset_index(level=0,inplace =True )
count_df.columns = ['ImageId','Ships']
temp = pd.merge(ship_df,count_df,how ='left',on ='ImageId')
final_count_df =  temp.fillna(0)
final_count_df = final_count_df.astype({'Ships': 'int32'})
final_count_df['Ships'].value_counts().plot(kind ='bar',title ='Ship Counts',figsize = (10,5))

# rebalancing train data
zero_ship_index = final_count_df.index[final_count_df['Ships'] ==0].tolist()
one_ship_index = final_count_df.index[final_count_df['Ships'] ==1].tolist()
zero_one_index = set(zero_ship_index).union(set(one_ship_index))
ship_index = list(set(range(192556)).difference(zero_one_index))

sampling_zero = random.sample(zero_ship_index, k =10000)
sampling_one = random.sample(one_ship_index, k =10000)
sample_zero_ship = final_count_df.iloc[sampling_zero,]
sample_one_ship = final_count_df.iloc[sampling_one,]
sample_ship = final_count_df.iloc[ship_index,]
ship_zero_one_df = pd.concat([sample_zero_ship,sample_one_ship])
ship_df = pd.concat([ship_zero_one_df,sample_ship])
ship_df = ship_df.reset_index(drop =True)

ship_df['Ships'].value_counts().plot(kind ='bar',title ='Ship Counts',figsize = (10,5))

train_df = ship_df.sample(n =1000)
train_df = train_df.reset_index(drop =True)

def rle_decode(mask, shape =(768, 768)): # function to decode rle_encoding
    if mask ==0:
        return np.zeros(shape)
    else:
        s = mask.split()
        starts, lengths = [np.asarray(x, dtype =int) for x in (s[0:][::2], s[1:][::2])]
        starts -=1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype =np.uint8)
        for start, end in zip(starts, ends):
            img[start:end] =1
        return img.reshape(shape).T #Modification of Direction

x = []
y = []

pbar=pyprind.ProgBar(1000)
for i in range(1000):
    img = cv2.imread(img_dir +'\\'+train_df['ImageId'][i])
    img = cv2.resize(img,(256,256))
    img_masks = seg.loc[seg['ImageId'] == train_df['ImageId'][i], 'EncodedPixels'].tolist()
    new_masks = np.zeros((768, 768))
    for mask in img_masks:
        new_masks += rle_decode(mask)
    new_masks = cv2.resize(new_masks,(256,256))
    x.append(img)
    y.append(new_masks)
    pbar.update()

x = np.array(x)
y = np.array(y)
x = x /255
y = y.reshape(y.shape[0],y.shape[1],y.shape[2],1)

## Building InceptionV3 + Resnet Model

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D,UpSampling2D, Dropout, concatenate, BatchNormalization, Flatten, Dense, add,Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow.keras.utils
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callback_list = [EarlyStopping(monitor ='val_loss',patience =3), ModelCheckpoint('Ship_detect.h5',monitor ='val_loss', save_best_only ='True')]

base_inception = InceptionV3(input_shape =  (256,256,3), include_top = False, weights ='imagenet')
base_inception.trainable = False
base_inception.summary()
concatenate0 = Model(inputs =base_inception.input,outputs =base_inception.layers[40].output).output
drop = Dropout(0.5)(concatenate0)

up1 =  Conv2DTranspose(512, 5, activation ='relu')(UpSampling2D(size =(2,2))(drop))
shortcut1 = up1
conv1 = Conv2D(512, 3, activation ='relu', padding ='same', kernel_initializer ='he_normal')(up1)
conv1 = Conv2D(512, 3, activation ='relu', padding ='same', kernel_initializer ='he_normal')(conv1)
conv1 = BatchNormalization()(conv1)
merge1 = add([shortcut1,conv1])

up2 = Conv2DTranspose(256, 5, activation ='relu')(UpSampling2D(size =(2,2))(merge1))
shortcut2 = up2
conv2 = Conv2D(256, 3, activation ='relu', padding ='same', kernel_initializer ='he_normal')(up2)
conv2 = Conv2D(256, 3, activation ='relu', padding ='same', kernel_initializer ='he_normal')(conv2)
conv2 = BatchNormalization()(conv2)
merge2 = add([shortcut2,conv2])

up3 = Conv2D(128, 2, activation ='relu', padding ='same', kernel_initializer ='he_normal')(UpSampling2D(size =(2,2))(merge2))
shortcut3 = up3
conv3 = Conv2D(128, 3, activation ='relu', padding ='same', kernel_initializer ='he_normal')(up3)
conv3 = Conv2D(128, 3, activation ='relu', padding ='same', kernel_initializer ='he_normal')(conv3)
conv3 = BatchNormalization()(conv3)
merge3 = add([shortcut3,conv3])

conv4 = Conv2D(64, 3, activation ='relu', padding ='same', kernel_initializer ='he_normal')(merge3)

img_output = Conv2D(1, 1, activation ='sigmoid')(conv4)

ship_detect = Model(inputs = base_inception.input, outputs = img_output)
ship_detect.summary()

from tensorflow.keras.utils import plot_model
plot_model(ship_detect, show_shapes=True)

for layer in ship_detect.layers[:41]:
  layer.trainable = False

ship_detect.compile(optimizer='Adam', loss ='binary_crossentropy', metrics = ['accuracy'])

history = ship_detect.fit(x,y,epochs =20,batch_size =1,validation_split =0.1,callbacks = callback_list)
