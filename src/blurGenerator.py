import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from PIL import Image
import keras.utils as image
import os,shutil
import random

"""
this class shoud get image library 
apply spatial blur on all images , and return (ax,ay)
acceleration in x, y axis foreach image

"""

class BlurGenerator:
    def __init__(self,path_to_images,NOF_train,NOF_Test) -> None:
        self.path_to_images=path_to_images
        self.img_to_ax_ay_dict = {}
        self.NOF_IMAGES = NOF_train + NOF_train
        self.img_stack = 0
        self.acc = 0
        self.IMAGE_LEN = 218
        self.IMAGE_WiDTH = 178
        self.num_time_samples = 120
        self.T = 15


    def transfer_function(self, u,v, a_x,a_y, t):
        """
        Calculates the transfer function H(u, v) for the blurring caused by uniform acceleration
        in the x direction during exposure.

        Parameters:
            u : numpy array
                Spatial frequency in the x direction.
            a : float
                Acceleration.
            t : numpy array
                Array of time points during exposure.

        Returns:
            H : numpy array
                Transfer function H(u, v).
        """
        H = np.exp(-1j * np.pi * u * a_x * t**2)*np.exp(-1j * np.pi * v * a_y * t**2)
        return H

    def apply_motion_blur(self,image, a_x, a_y, t, cutoff_frequency, num_time_samples):
        """
        Applies motion blur to the input image using the specified acceleration and exposure time,
        integrates over time, and applies a low-pass filter before inverse Fourier transform reconstruction.

        Parameters:
            image : numpy array
                Input image (grayscale).
            a : float
                Acceleration.
            t : numpy array
                Array of time points during exposure.
            cutoff_frequency : float
                Cutoff frequency for the low-pass filter.
            num_time_samples : int
                Number of time samples for integration.

        Returns:
            blurred_image : numpy array
                Blurred image with temporal blur and low-pass filtering applied.
        """
        # Convert image to float32
        image_float = image.astype(np.float32)

        # Compute Fourier transform of image
        F = fft2(image_float)

        # Get dimensions of image
        h, w = image_float.shape

        # Create meshgrid of spatial frequencies
        u = np.fft.fftfreq(w)
        v = np.fft.fftfreq(h)
        u_mesh, v_mesh = np.meshgrid(u, v, indexing='xy')

        # Initialize transfer function
        H_sum = np.zeros_like(u_mesh, dtype=np.complex64)

        # Integrate over time
        for i in range(num_time_samples):
            # Calculate transfer function for current time sample
            H = self.transfer_function(u_mesh,v_mesh, a_x, a_y, t[i])
            # Accumulate transfer function over time
            H_sum += H

        # Average transfer function over time
        #H_avg = H_sum / num_time_samples
        H_avg = H_sum 

        # Apply low-pass filter in frequency domain
        low_pass_filter = np.exp(-(u_mesh**2 + v_mesh**2) / (2 * (cutoff_frequency**2)))
        #H_filtered = H_avg * low_pass_filter
        H_filtered = H_avg 

        # Apply transfer function
        F_blurred = F * H_filtered

        # Inverse Fourier transform
        blurred_image = np.real(ifft2(F_blurred))

        return blurred_image , H_filtered


    def RandomBlureImage(self,image):
        a_x = random.random()*2
        a_y = random.random()*2
        # Parameters
        a = 0.9  # Acceleration
        #T = 15.0  # Total exposure time
        num_time_samples = 120  # Number of time samples for integration
        cutoff_frequency = 0.5  # Cutoff frequency for low-pass filter

        # Generate array of time points during exposure
        t = np.linspace(0, self.T, num_time_samples)

        blurred_image, Distortion_filter = self.apply_motion_blur(image, a_x,a_y, t, cutoff_frequency, num_time_samples)
        
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 3, 1)
        # plt.imshow(image, cmap='gray')
        # plt.title('Original Image')
        # plt.axis('off')

        # plt.subplot(1, 3, 2)
        # plt.imshow(blurred_image, cmap='gray')
        # plt.title('spacial accelerated Blur image a_x =' +str(a_x)+ 'a_y =' +str(a_y))
        # plt.axis('off')

        # plt.subplot(1, 3, 3)
        # plt.imshow(np.real(ifft2(fft2(blurred_image)/Distortion_filter)), cmap='gray')
        # plt.title('Restored image')
        # plt.axis('off')

        # plt.show()
        return a_x,a_y,blurred_image

    def CreateDataSet(self):
        #open images
        images_pool = [os.path.join( self.path_to_images, fname) for fname in os.listdir( self.path_to_images)]
        #img = image.load_img(images_pool[3])
        self.img_stack = np.empty((0,218,178))
        self.acc = np.empty((0,2))
        for img in images_pool[0:self.NOF_IMAGES]:
            image = np.array(Image.open(img).convert('L'))
            image_float = image.astype(np.float32)/255
            #plt.imshow(image_float)
            a_x,a_y,blure_image = self.RandomBlureImage(image_float)
            #self.img_to_ax_ay_dict[blure_image] = [a_x,a_y]
            self.img_stack = np.vstack((self.img_stack,blure_image[np.newaxis,:,:]))
            self.acc = np.vstack((self.acc,[a_x,a_y]))
        #blue images
        plt.figure(figsize=(10, 5))
        plt.imshow(image_float, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        self.GetRestoredImage(blure_image/255,a_x,a_y)

        return self.img_stack , self.acc
        #return ax,ay,original_img , blur_img OR dictionary blur_img -> (ax,ay,original_img)

    def GetRestoredImage(self,blure_image,a_x,a_y):
        h, w = blure_image.shape

        # Create meshgrid of spatial frequencies
        u = np.fft.fftfreq(w)
        v = np.fft.fftfreq(h)
        u_mesh, v_mesh = np.meshgrid(u, v, indexing='xy')

        # Initialize transfer function
        H_sum = np.zeros_like(u_mesh, dtype=np.complex64)
        #T = 15
        t = np.linspace(0, self.T, self.num_time_samples)
        # Integrate over time
        for i in range(self.num_time_samples):
            # Calculate transfer function for current time sample
            H = self.transfer_function(u_mesh,v_mesh, a_x, a_y, t[i])
            # Accumulate transfer function over time
            H_sum += H

        # Average transfer function over time
        #H_avg = H_sum / num_time_samples
        H_avg = H_sum 

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(np.real(ifft2(fft2(blure_image)/H_avg)), cmap='gray')
        plt.title('Restored image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(blure_image, cmap='gray')
        plt.title('Blur Image')
        plt.axis('off')

        