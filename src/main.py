from blurGenerator import BlurGenerator
import matplotlib.pyplot as plt
import keras.utils as image
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout ,MaxPooling2D,Conv2D,Flatten,Dense,concatenate
from keras.models import load_model
import numpy as np
from tensorflow.keras import Input , Model
from enum import Enum

NOF_train = 150
class Mode(Enum):
    CONVNET = 1
    CONCATENATED_FULLY_CONNECTED_CONVNET = 2

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

def buildConcatenatedNetwork(blurGenInstance):
        #build fullyConnected for Singular values
        FC_inputs = Input(shape=(blurGenInstance.IMAGE_WiDTH),name="SVD_inputs")
        layer1 = Dense(units = 800,activation = 'relu')(FC_inputs) 
        layer1 = Dropout(0.1)(layer1)
        layer1 = Dense(units=500, activation='relu')(layer1)
        layer1 = Dropout(0.1)(layer1)
        layer1 = Dense(units  = 128, activation='sigmoid')(layer1)
        layer1 = Dropout(0.1)(layer1)
        output_Fc= Dense(128,activation='sigmoid')(layer1)
        #build convnet for blur images
        convnet_input = Input(shape=(blurGenInstance.IMAGE_LEN, blurGenInstance.IMAGE_WiDTH, 1),name="Convnet_inputs")
        layer2 = Conv2D(32, (3, 3), activation='relu')(convnet_input)
        layer2 = MaxPooling2D((2, 2))(layer2)
        layer2 = Conv2D(64, (3, 3), activation='relu')(layer2)
        layer2 = MaxPooling2D((2, 2))(layer2)
        layer2 = Conv2D(128, (3, 3), activation='relu')(layer2)
        layer2 = MaxPooling2D((2, 2))(layer2)
        layer2 = Conv2D(128, (3, 3), activation='relu')(layer2)
        layer2 = MaxPooling2D((2, 2))(layer2)
        layer2 = Flatten()(layer2)
        layer2 = Dense(512, activation='sigmoid')(layer2)
        layer2 = Dropout(0.3)(layer2)
        output_convnet =  Dense(128, activation='sigmoid')(layer2)
        #concatenate layers
        concatenated = concatenate([output_Fc, output_convnet]) 
        output = Dense(2, activation='linear')(concatenated)
        model = Model(inputs=[FC_inputs,convnet_input],outputs=output)
        model.compile(optimizer='adam',  loss='mean_squared_error', metrics=['mae'])
        return model

def getNormalizeDataSet(ims_stack,acc):
        return ims_stack[:NOF_train].astype("float32")/255, acc[:NOF_train], ims_stack[NOF_train:].astype("float32")/255 ,acc[NOF_train:]

def generateSVDdataset(img_stack,acc,blurInstance):
     S_stack = np.empty((0,blurInstance.IMAGE_WiDTH))
     for img in img_stack:
        U, S, VT = np.linalg.svd(img, full_matrices=False)
        S_stack=np.vstack((S_stack,S))
     return S_stack[:NOF_train].astype("float32")/(255*255), acc[:NOF_train], S_stack[NOF_train:].astype("float32")/(255*255) ,acc[NOF_train:]
     
blurGenInstance = BlurGenerator('celebA_\\img_align_celeba\\img_align_celeba')
img_stack,acc = blurGenInstance.CreateDataSet()
#Generate SVD Dataset
S_stack_train , S_acc_train,S_stack_test,S_acc_test = generateSVDdataset(img_stack,acc,blurGenInstance)
#Generate blur images Dataset with accelerations(x,y)
train_data, train_label ,test_data, test_label = getNormalizeDataSet(img_stack,acc)



skip_training = 0
mode = Mode.CONCATENATED_FULLY_CONNECTED_CONVNET
if(mode==Mode.CONCATENATED_FULLY_CONNECTED_CONVNET):
    if skip_training:
        concatenated_model = load_model('Restore_blure_images_concatenated.keras')
        result =  concatenated_model.predict(test_data)
    else:
        #fit SVD fully connected model
        concatenated_model = buildConcatenatedNetwork(blurGenInstance)
        concatenated_history = concatenated_model.fit([S_stack_train,train_data], S_acc_train, epochs=85, validation_split=0.2, verbose=1)
        concatenated_result =  concatenated_model.predict([S_stack_test,test_data])
        concatenated_model.save('Restore_blure_images_concatenated.keras')

if(mode==Mode.CONVNET):
    if skip_training:
        model = load_model('Restore_blure_images.keras')
        result =  model.predict(test_data)
    else:
        #fit blur image model
        model = build_model(blurGenInstance)
        history = model.fit(train_data, train_label, epochs=85, validation_split=0.2, verbose=1)
        result =  model.predict(test_data)
        model.save('Restore_blure_images.keras')

for i in range(10):
    blurGenInstance.GetRestoredImage(test_data[i],concatenated_result[i][0],concatenated_result[i][1])
#in step 2 evolse S from svd decomposition