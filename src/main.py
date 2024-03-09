from blurGenerator import BlurGenerator
import matplotlib.pyplot as plt
import keras.utils as image
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout ,MaxPooling2D,Conv2D,Flatten,Dense
from keras.models import load_model

NOF_train = 150

def build_model(blurGenInstance):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(blurGenInstance.IMAGE_LEN, blurGenInstance.IMAGE_WiDTH, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

def getNormalizeDataSet(ims_stack,acc):
        return ims_stack[:NOF_train].astype("float32")/255, acc[:NOF_train], ims_stack[NOF_train:].astype("float32")/255 ,acc[NOF_train:]

blurGenInstance = BlurGenerator('celebA_\\img_align_celeba\\img_align_celeba')
ims_stack,acc = blurGenInstance.CreateDataSet()

train_data, train_label ,test_data, test_label = getNormalizeDataSet(ims_stack,acc)

skip_training = 1
if skip_training:
    model = load_model('Restore_blure_images.keras')
    result =  model.predict(test_data)
else:
    model = build_model(blurGenInstance)
    history = model.fit(train_data, train_label, epochs=85, validation_split=0.2, verbose=1)
    result =  model.predict(test_data)
    model.save('Restore_blure_images.keras')

for i in range(10):
    blurGenInstance.GetRestoredImage(test_data[10],result[10][0],result[10][1])
#in step 2 evolse S from svd decomposition