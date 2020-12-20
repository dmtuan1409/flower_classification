import numpy as np
import csv
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization,Conv2D, MaxPooling2D,Concatenate, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Xem mỗi hoa có bao nhiêu ảnh
data_folder="Flower/Flower"
arrImg=[]
nameFlower=[]
for folder in os.listdir(data_folder):
        nameFlower.append(folder)
        curren_path = os.path.join(data_folder,folder)
        sum=0
        for i in os.listdir(curren_path):
            sum=sum+1
        arrImg.append(sum)
        print("Hoa "+folder+" có "+str(sum)+" ảnh")
print(arrImg)
print(nameFlower)

#Đọc ảnh và thêm vào mảng, thêm nhãn tương ứng vào mảng
image=[]
label=[]
for folder in os.listdir(data_folder):
        curren_path = os.path.join(data_folder,folder)
        sum=0
        for i in os.listdir(curren_path):
                img_path = os.path.join(curren_path,i)
                img = cv2.imread(img_path, 1)
                image.append(img)
                label.append(folder)
                sum=sum+1
                print("anh thu "+str(sum)+ " folder "+ folder)
print(len(image))
print(len(label))

#Chuyển nhãn từ dạng chuỗi sang số từ 0-23
classes=["Black eyed susan","Bluebell","Buttercup","Chrysanthemum","Colstfoot","Cornflower","Cowslip","Crocus","Daffodil","Daisy","Dandelion","Hibiscus","Iris","Lavender","LilyValey","Lotus","Orchid","Pansy","Snowdrop","Sunflowers","Tigerlily","Tulips","Windflower","Rose"]
for i in range(len(classes)):
        for j in range(len(label)):
                if classes[i] == label[j]:
                        print(label[j])
                        label[j] = i
                        print(label[j])

#Demo ảnh hoa, số lượng từng loại hoa
plt.imshow(image[12])
plt.show()
plt.figure(figsize=(32,10))
plt.bar(nameFlower,arrImg,color="green")
plt.title("Biểu đồ cột số lượng từng loài hoa")
plt.xlabel("Tên hoa")
plt.ylabel("Số lượng")
plt.show()

#Chuyển thành numpy array, chuyển sang float16 và chuẩn hóa về (0,1)
image = np.array(image).astype("float16")
image = image/255
label = np.array(label).astype("uint8")
print(np.shape(image))
print(np.shape(label))

#Chia train, test
X_train,X_val,y_train,y_val=train_test_split(image,label,test_size=0.3, random_state=42)

#Reshape va ma hoa onehot
X_train = X_train.reshape(X_train.shape[0],180,180,3)
X_val = X_val.reshape(X_val.shape[0],180,180,3)
y_train = tf.keras.utils.to_categorical(y_train, 24)
y_val =  tf.keras.utils.to_categorical(y_val, 24)

#Xây dựng mô hình
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(180, 180, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(24, activation='softmax'))
model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer='adam', metrics=['accuracy', 'Precision', 'Recall'])

#Xem thông tin mô hình
model.summary()

#Data augmentation
aug_train = ImageDataGenerator(rotation_range=0.18, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

#Training
epochs=1000
batch_size=64
H=model.fit_generator(aug_train.flow(X_train,y_train,batch_size=batch_size),validation_data=(X_val,y_val),steps_per_epoch=X_train.shape[0]//batch_size,epochs=epochs,verbose=1)

#Save mô hình
model.save("flowerclassification.h5")

#In ra ảnh quá trình training, file model.png
fig = plt.figure()
numOfEpoch = 1000
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numOfEpoch), H.history['Precision'], label='training Precision')
plt.plot(np.arange(0, numOfEpoch), H.history['val_Precision'], label='validation Precision')
plt.plot(np.arange(0, numOfEpoch), H.history['Recall'], label='training Recall')
plt.plot(np.arange(0, numOfEpoch), H.history['val_Recall'], label='validation Recall')
plt.plot(np.arange(0, numOfEpoch), H.history['accuracy'], label='accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy|Precision|Recall')
plt.legend()