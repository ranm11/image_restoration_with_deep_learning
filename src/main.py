from blurGenerator import BlurGenerator
import matplotlib.pyplot as plt
import keras.utils as image
from keras.models import load_model
import numpy as np
from DeepLearningModels import deepLearningModels 
from enum import Enum
import cv2
from keras.utils import plot_model

class Mode(Enum):
    CONVNET = 1
    CONCATENATED_FULLY_CONNECTED_CONVNET = 2
    SOBEL = 3
    CONVNET_COVARIANCE = 4
    CONVNET_OVER_SOBEL = 5

NOF_train = 150
NOF_test = 50
skip_training = 0
mode = Mode.CONVNET_OVER_SOBEL


def getNormalizeDataSet(ims_stack,acc):
        return ims_stack[:NOF_train].astype("float32")/255, acc[:NOF_train], ims_stack[NOF_train:].astype("float32")/255 ,acc[NOF_train:]

def generateSVDdataset(img_stack,acc,blurInstance):
     S_stack = np.empty((0,blurInstance.IMAGE_WiDTH))
     for img in img_stack:
        U, S, VT = np.linalg.svd(img, full_matrices=False)
        S_stack=np.vstack((S_stack,S))
     return S_stack[:NOF_train].astype("float32")/(255*255), acc[:NOF_train], S_stack[NOF_train:].astype("float32")/(255*255) ,acc[NOF_train:]

def  getSobelDataset(img_stack,acc):
    sobel_img_stack_x = np.empty((0,218,178))
    sobel_img_stack_y = np.empty((0,218,178))
    for img in img_stack:
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobelx = cv2.Sobel(sobelx, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(sobely, cv2.CV_64F, 0, 1, ksize=3)
        # Compute the covariance matrix of gradients
        show_sobel_img = 0
        if show_sobel_img:
            plt.figure(figsize=(10, 5))
            plt.subplot(131)
            plt.imshow(sobelx, cmap='gray')
            plt.title('sobel2_x')
            plt.subplot(132)
            plt.imshow(sobely, cmap='gray')
            plt.title('sobel3_y')
            plt.subplot(133)
            plt.imshow(img, cmap='gray')
            plt.title('blurd image')
            plt.axis('off')

        #grad_matrix = np.column_stack((grad_x_flatten, grad_y_flatten))
        #cov_matrix = np.cov(grad_matrix.T)
        sobel_img_stack_x = np.vstack((sobel_img_stack_x,sobelx[np.newaxis,:,:]))
        sobel_img_stack_y = np.vstack((sobel_img_stack_y,sobely[np.newaxis,:,:]))

    return sobel_img_stack_x[:NOF_train].astype("float32")/255 ,sobel_img_stack_y[:NOF_train].astype("float32")/255, acc[:NOF_train].astype("float32") ,sobel_img_stack_x[NOF_train:].astype("float32")/255,sobel_img_stack_y[NOF_train:].astype("float32")/255, acc[NOF_train:].astype("float32")         

def  getCovarianceMatrixDataset(img_stack,acc):
    covariance_stack = np.empty((0,4))
    for img in img_stack:
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobelx = cv2.Sobel(sobelx, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(sobely, cv2.CV_64F, 0, 1, ksize=3)
        # Compute the covariance matrix of gradients
        show_sobel_img = 0
        if show_sobel_img:
            plt.figure(figsize=(10, 5))
            plt.subplot(131)
            plt.imshow(sobelx, cmap='gray')
            plt.title('sobel_x')
            plt.subplot(132)
            plt.imshow(sobely, cmap='gray')
            plt.title('sobel_y')
            plt.subplot(133)
            plt.imshow(img, cmap='gray')
            plt.title('blurd image')
            plt.axis('off')

        grad_x_flatten = sobelx.flatten()
        grad_y_flatten = sobely.flatten()
        grad_matrix = np.column_stack((grad_x_flatten, grad_y_flatten))
        cov_matrix = np.cov(grad_matrix.T)
        covariance_stack=np.vstack((covariance_stack,np.reshape(cov_matrix,(4))))

    return covariance_stack[:NOF_train].astype("float32")/255 , acc[:NOF_train].astype("float32") ,covariance_stack[NOF_train:].astype("float32")/255, acc[NOF_train:].astype("float32")         

blurGenInstance = BlurGenerator('celebA_\\img_align_celeba\\img_align_celeba',NOF_train,NOF_test)
dl_Instance = deepLearningModels()
img_stack,acc = blurGenInstance.CreateDataSet()
#Generate SVD Dataset
S_stack_train , S_acc_train,S_stack_test,S_acc_test = generateSVDdataset(img_stack,acc,blurGenInstance)
#Generate blur images Dataset with accelerations(x,y)
train_data, train_label ,test_data, test_label = getNormalizeDataSet(img_stack,acc)


if(mode == Mode.CONVNET_OVER_SOBEL):
    train_sobel_x ,train_sobel_y, train_label ,test_sobel_x,test_sobel_y, test_label = getSobelDataset(img_stack,acc)             
    if skip_training:
        sobel_convnet_model_X = load_model('sobel2_x_model.keras')
        sobel_convnet_model_Y = load_model('sobel2_y_model.keras')
        result_x =  sobel_convnet_model_X.predict(test_sobel_x)
        result_y =  sobel_convnet_model_Y.predict(test_sobel_y)
        result = np.concatenate((result_x,result_y),axis=1)
        np.mean(np.abs(result-test_label))
        
    else:    
        sobel_convnet_model_X = dl_Instance.build_convnet_over_sobel_model(blurGenInstance)
        sobel_convnet_model_Y = dl_Instance.build_convnet_over_sobel_model(blurGenInstance)
        history_x = sobel_convnet_model_X.fit(train_sobel_x, train_label[:,0].reshape(NOF_train,1), epochs=85, validation_split=0.2, verbose=1)
        history_y = sobel_convnet_model_Y.fit(train_sobel_y, train_label[:,1].reshape(NOF_train,1), epochs=85, validation_split=0.2, verbose=1)
        result_x =  sobel_convnet_model_X.predict(test_sobel_x)
        result_y =  sobel_convnet_model_Y.predict(test_sobel_y)
        result = np.concatenate((result_x,result_y),axis=1)
        np.mean(np.abs(result-test_label))
        dl_Instance.plotLoss(history_x)
        dl_Instance.plotLoss(history_y)
        sobel_convnet_model_X.save('sobel2_x_model.keras')
        sobel_convnet_model_Y.save('sobel2_y_model.keras')
        plot_model(sobel_convnet_model_X, to_file='Sobel2_model_plot.png', show_shapes=True, show_layer_names=True)

if(mode==Mode.CONCATENATED_FULLY_CONNECTED_CONVNET):
    if skip_training:
        concatenated_model = load_model('Restore_blure_images_concatenated.keras')
        result =  concatenated_model.predict([S_stack_test,test_data])
        plot_model(concatenated_model, to_file='SVD_CONVNET_concatenated_model_plot.png', show_shapes=True, show_layer_names=True)
        dl_Instance.plotLoss(result)
        np.mean(result-test_label) # 8% deviation 

    else:
        #fit SVD fully connected model
        concatenated_model = dl_Instance.buildConcatenatedNetwork(blurGenInstance)
        concatenated_history = concatenated_model.fit([S_stack_train,train_data], S_acc_train, epochs=85, validation_split=0.2, verbose=1)
        concatenated_result =  concatenated_model.predict([S_stack_test,test_data])
        concatenated_model.save('Restore_blure_images_concatenated.keras')

if(mode==Mode.CONVNET):
    if skip_training:
        conv_model = load_model('Restore_blure_images.keras')
        result =  conv_model.predict(test_data)
        dl_Instance.plotLoss(conv_model.history)
        plot_model(conv_model, to_file='convNet_model_plot.png', show_shapes=True, show_layer_names=True)
        np.mean(np.abs(result-test_label))
    else:
        #fit blur image model
        conv_model = dl_Instance.build_convnet_model(blurGenInstance)
        history = conv_model.fit(train_data, train_label, epochs=85, validation_split=0.2, verbose=1)
        dl_Instance.plotLoss(history)
        result =  conv_model.predict(test_data)
        conv_model.save('Restore_blure_images.keras')
        #
        # conv_model.save('ConvNet_300_images.keras')
        
        np.mean(np.abs(result-test_label))
        training_accuracy = history.history['mae']
        plt.figure(figsize=(10, 5))
        plt.plot(training_accuracy )
        validation_accuracy = history.history['val_mae']
        plt.plot(validation_accuracy )
if(mode==Mode.SOBEL):
        train_cov , train_label ,test_cov, test_label = getCovarianceMatrixDataset(img_stack,acc)        
        
        if skip_training:
            covariance_model = load_model('Restore_blure_images_covariance_sobel_2.keras')
            result =  covariance_model.predict(test_cov)
            plot_model(covariance_model, to_file='Sobel_model_plot.png', show_shapes=True, show_layer_names=True)
            np.mean(np.abs(result-test_label))
        else:
            model = dl_Instance.build_covariance_analyzer_model(blurGenInstance)
            history = model.fit(train_cov, train_label, epochs=85, validation_split=0.2, verbose=1)
            result =  model.predict(test_cov)
            dl_Instance.plotLoss(history)
            model.save('Restore_blure_images_covariance_sobel_2.keras')
            np.mean(np.abs(result-test_label))
            
        # Print dominant direction and corresponding eigenvalue
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 3, 1)
        # plt.imshow(sobelx, cmap='gray')
        # plt.subplot(1, 3, 2)
        # plt.imshow(sobely, cmap='gray')
        # plt.subplot(1, 3, 3)
        # plt.imshow(train_data[0], cmap='gray')
        #give your system sobelx to predict x a_x and sobely to predict a_y
        #try to predict from cov matrix
if(mode==Mode.CONVNET_COVARIANCE):
        
        train_cov , train_label ,test_cov, test_label = getCovarianceMatrixDataset(img_stack,acc)             
    
        if skip_training:
            convnet_covariance_model = load_model('Restore_blure_images_convnet_covariance_concatenated.keras')
            plot_model(convnet_covariance_model, to_file='convNet_Covariance_input_model_plot.png', show_shapes=True, show_layer_names=True)
            result =  convnet_covariance_model.predict([test_cov,test_data])
            np.mean(result-test_label) # 8% deviation 

        else:
            
            convnet_covariance_model = dl_Instance.build_convnet_covariance_concatenated_model(blurGenInstance)
            history = convnet_covariance_model.fit([train_cov,train_data], S_acc_train, epochs=85, validation_split=0.2, verbose=1)
            result =  convnet_covariance_model.predict([test_cov,test_data])
            convnet_covariance_model.save('Restore_blure_images_convnet_covariance_concatenated.keras')
            dl_Instance.plotLoss(history)
        np.mean(np.abs(result-test_label)) 
        training_accuracy = history.history['mae']
        plt.figure(figsize=(10, 5))
        plt.plot(training_accuracy )
        validation_accuracy = history.history['val_mae']
        plt.plot(validation_accuracy )
for i in range(15):
    #blurGenInstance.GetRestoredImage(test_data[i],concatenated_result[i][0],concatenated_result[i][1])
    blurGenInstance.GetRestoredImage(test_data[i],result[i][0],result[i][1])
     
#in step 2 evolse S from svd decomposition 48 47 44 42 41 39 38
    

    #show what happen if deviate in a few 
    #for sobel tecknique
    # mean deviavion of 3%