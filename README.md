Image Deconvolution is a well known technique to restore blur images within 
Its been used for image reconstruction in astronomy and medical imaging.
This procedure works perfectly provided we know the accelerations vectors that the image has gone through, (Ax,Ay).
We simply employ a reconstruction formula that works like a magic.
![image](https://github.com/user-attachments/assets/4a131fba-3beb-4956-9d5c-d58a82eb0de6)

But what if we don’t know (Ax,Ay) acceleration components ? this is the place where Deep Learning take over !
This workshop try to predict the (Ax,Ay) parameters and employ the reconstruction formula within the predicted values in several approaches.

