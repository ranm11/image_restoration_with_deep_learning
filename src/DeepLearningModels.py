from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout ,MaxPooling2D,Conv2D,Flatten,Dense,concatenate
from tensorflow.keras import Input , Model
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

class deepLearningModels:
    def __init__(self) -> None:
        self.zevel=0

    def plotLoss(self,history):
        plt.figure()
        plt.subplot(211)
        loss = history.history['loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.title('Training loss')
        plt.subplot(212)
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, val_loss, 'bo', label='Training loss')
        plt.title('validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def build_convnet_covariance_concatenated_model(self,blurGenerator):
        FC_inputs = Input(shape={4},name="SVD_inputs")
        layer1 = Dense(units = 80,activation = 'relu')(FC_inputs) 
        layer1 = Dropout(0.1)(layer1)
        layer1 = Dense(units=50, activation='relu')(layer1)
        layer1 = Dropout(0.1)(layer1)
        layer1 = Dense(units  = 18, activation='sigmoid')(layer1)
        layer1 = Dropout(0.1)(layer1)
        output_Fc= Dense(2,activation='linear')(layer1)
        convnet_input = Input(shape=(blurGenerator.IMAGE_LEN, blurGenerator.IMAGE_WiDTH, 1),name="Convnet_inputs")
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
        model.compile(optimizer='adam',  loss='mean_squared_error', metrics=['accuracy'])
        return model

    def build_covariance_analyzer_model(self,blurGenInstance):
        FC_inputs = Input(shape={4},name="SVD_inputs")
        layer1 = Dense(units = 80,activation = 'relu')(FC_inputs) 
        layer1 = Dropout(0.1)(layer1)
        layer1 = Dense(units=50, activation='relu')(layer1)
        layer1 = Dropout(0.1)(layer1)
        layer1 = Dense(units  = 18, activation='sigmoid')(layer1)
        layer1 = Dropout(0.1)(layer1)
        output_Fc= Dense(2,activation='linear')(layer1)
        model = Model(inputs=FC_inputs,outputs=output_Fc)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        return model

    def build_convnet_model(self,blurGenInstance, weight_decay = 1e-4):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(blurGenInstance.IMAGE_LEN, blurGenInstance.IMAGE_WiDTH, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        #model.add(Dropout(0.2))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (2, 2),kernel_regularizer=l2(weight_decay), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        return model

    def build_convnet_over_sobel_model(self,blurGenInstance, weight_decay = 1e-4):
            model = Sequential()
            model.add(Conv2D(32, (8, 8),kernel_regularizer=l2(weight_decay), activation='relu', input_shape=(blurGenInstance.IMAGE_LEN, blurGenInstance.IMAGE_WiDTH, 1)))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (5, 5),kernel_regularizer=l2(weight_decay), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(128, (3, 3),kernel_regularizer=l2(weight_decay), activation='relu'))
            #model.add(Dropout(0.2))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(128, (3, 3),kernel_regularizer=l2(weight_decay), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='linear'))
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
            return model

    def buildConcatenatedNetwork(self,blurGenInstance):
            #build fullyConnected for Singular values
            FC_inputs = Input(shape={blurGenInstance.IMAGE_WiDTH},name="SVD_inputs")
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

    



