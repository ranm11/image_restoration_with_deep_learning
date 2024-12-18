Image Deconvolution is a well known technique to restore blur images affected by lateral movement.
Its been widely  used for image reconstruction in astronomy and medical imaging.
This procedure works perfectly provided we know the accelerations vectors that the image has gone through, (Ax,Ay) 
Within the below procedure:


![image](https://github.com/user-attachments/assets/9d0c07a5-b00e-40ad-b275-c4882859653d)



where Fr is the recontructed image.

We simply employ the above reconstruction formula on the blur image and it works like a magic.
![image](https://github.com/user-attachments/assets/4a131fba-3beb-4956-9d5c-d58a82eb0de6)

But what if we don’t know (Ax,Ay) acceleration components ? this is the place where Deep Learning take over !
This workshop try to predict the (Ax,Ay) parameters and employ the reconstruction formula within the predicted values in several approaches.

Final Results can be seen below :

Restoring accelaration parameters with Deep Learning and restore images accordingly 

Enjoy !

![image](https://github.com/user-attachments/assets/c67547b5-e052-4849-a0f1-a326a552a7ed)



