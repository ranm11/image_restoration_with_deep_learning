from blurGenerator import BlurGenerator
import matplotlib.pyplot as plt
import keras.utils as image
from keras.models import load_model
import numpy as np
from DeepLearningModels import deepLearningModels 
from enum import Enum
import cv2


class Mode(Enum):
    CONVNET = 1
    CONCATENATED_FULLY_CONNECTED_CONVNET = 2
    SOBEL = 3
    CONVNET_COVARIANCE = 4

NOF_train = 250
NOF_test = 50
skip_training = 0
mode = Mode.SOBEL


def getNormalizeDataSet(ims_stack,acc):
        return ims_stack[:NOF_train].astype("float32")/255, acc[:NOF_train], ims_stack[NOF_train:].astype("float32")/255 ,acc[NOF_train:]

def generateSVDdataset(img_stack,acc,blurInstance):
     S_stack = np.empty((0,blurInstance.IMAGE_WiDTH))
     for img in img_stack:
        U, S, VT = np.linalg.svd(img, full_matrices=False)
        S_stack=np.vstack((S_stack,S))
     return S_stack[:NOF_train].astype("float32")/(255*255), acc[:NOF_train], S_stack[NOF_train:].astype("float32")/(255*255) ,acc[NOF_train:]

def  getCovarianceMatrixDataset(img_stack,acc):
    covariance_stack = np.empty((0,4))
    for img in img_stack:
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        # Compute the covariance matrix of gradients
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




if(mode==Mode.CONCATENATED_FULLY_CONNECTED_CONVNET):
    if skip_training:
        concatenated_model = load_model('Restore_blure_images_concatenated.keras')
        result =  concatenated_model.predict([S_stack_test,test_data])
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
        np.mean(result-test_label) # 12% deviation 

    else:
        #fit blur image model
        conv_model = dl_Instance.build_model(blurGenInstance)
        history = conv_model.fit(train_data, train_label, epochs=85, validation_split=0.2, verbose=1)
        result =  conv_model.predict(test_data)
        conv_model.save('Restore_blure_images.keras')
        np.mean(result-test_label)/2 # 4% deviation 
        training_accuracy = history.history['mae']
        plt.figure(figsize=(10, 5))
        plt.plot(training_accuracy )
        validation_accuracy = history.history['val_mae']
        plt.plot(validation_accuracy )
if(mode==Mode.SOBEL):
        train_cov , train_label ,test_cov, test_label = getCovarianceMatrixDataset(img_stack,acc)        
        # Compute eigenvalues and eigenvectors of the covariance matrix
        # eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
# 
        #Sort eigenvalues and corresponding eigenvectors
        # sorted_indices = np.argsort(eigenvalues)[::-1]
        # eigenvalues_sorted = eigenvalues[sorted_indices]
        # eigenvectors_sorted = eigenvectors[:, sorted_indices]
# 
        #Extract dominant direction (eigenvector corresponding to largest eigenvalue)
        # dominant_direction = eigenvectors_sorted[:, 0]
        if skip_training:
            covariance_model = load_model('Restore_blure_images_covariance.keras')
            result =  covariance_model.predict(test_cov)
            np.mean(result-test_label)/2 # 3% precision 
        else:
            model = dl_Instance.build_covariance_analyzer_model(blurGenInstance)
            history = model.fit(train_cov, train_label, epochs=85, validation_split=0.2, verbose=1)
            result =  model.predict(test_cov)
            model.save('Restore_blure_images_covariance.keras')

        np.mean(result-test_label)/2 # 3% deviation 
        training_accuracy = history.history['accuracy']
        plt.figure(figsize=(10, 5))
        plt.plot(training_accuracy )
        validation_accuracy = history.history['val_accuracy']
        plt.plot(validation_accuracy )
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
            result =  convnet_covariance_model.predict([test_cov,test_data])
            np.mean(result-test_label) # 8% deviation 

        else:
            #fit SVD fully connected model
            convnet_covariance_model = dl_Instance.build_convnet_covariance_concatenated_model(blurGenInstance)
            history = convnet_covariance_model.fit([train_cov,train_data], S_acc_train, epochs=85, validation_split=0.2, verbose=1)
            result =  convnet_covariance_model.predict([test_cov,test_data])
            convnet_covariance_model.save('Restore_blure_images_convnet_covariance_concatenated.keras')
        np.mean(result-test_label)/2 # 4% deviation 
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