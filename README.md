# GestureRecognition

Hand gesture recognition & hand tracking script (python, pytorch), tweaked in this case, to be used as a controller for first person shooter game Call of Duty : Modern Warfare 2.

Hand gesture recognition is performed using deep learning model : VGG16 in PyTorch. A pretrained model is trained on hand gesture recognition dataset from kaggle (https://www.kaggle.com/gti-upm/leapgestrecog). Model was trained to classify between three different gestures : 'Palm', 'Fist' and 'Other' (I,C,index). In my case, training the model for 3 epochs was sufficient. Detected gestures were linked with simulating specific mouse action such as 'Fist' for left click and 'Other' for right click.

For hand tracking, OpenCV is used to perform contour detection based on colour (skin tone). Center of this detected contour is used for mouse movements.
