# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 18:10:44 2022

@author: Marina
"""

import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2


from tensorflow.keras.layers import BatchNormalization
import os
import seaborn as sns
from keras.applications.vgg16 import VGG16

###################################1#################################
# Read input images and assign labels based on folder names
print(os.listdir("D:/FCAI/Year (3)/S1/Artificial intelligence/AI"))

#Resize images
size = 256  

#Capture training data and labels into respective lists
train_images = []
train_labels = [] 
#######################################2#############################
for directory_path in glob.glob("D:/FCAI\Year (3)/S1/Artificial intelligence/AI\datasets of AI/Testing/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (size, size))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)

#Convert lists to arrays        
train_images = np.array(train_images)
train_labels = np.array(train_labels)
#############################################3####################################

# Capture test/validation data and labels into respective lists

test_images = []
test_labels = [] 

for directory_path in glob.glob("D:/FCAI/Year (3)/S1/Artificial intelligence/AI/datasets of AI/Testing/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (size, size))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(fruit_label)

#Convert lists to arrays                
test_images = np.array(test_images)
test_labels = np.array(test_labels)
###########################################4##############################
#Encode labels from text to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
# le.fit(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)
test_labels_encoded = le.transform(test_labels)

#Split data into test and train datasets (already split but assigning to meaningful convention)
x_train, y_train = train_images, train_labels_encoded 
x_test, y_test = test_images, test_labels_encoded
######################################5##################################

# Normalize pixel values to between 0 and 1
x_train = x_train / 255.0 
x_test = x_test / 255.0
######################################6###############################
#One hot encode y values for neural network. 
from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
##############################################7#######################

#Load model wothout classifier/fully connected layers
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(size, size, 3))
#include_top=False means : do not include dense layer 

#Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG_model.layers:
	layer.trainable = False

#Trainable parameters will be 0    
VGG_model.summary()  

# Testing
#Now, let us use features from convolutional network for RF
feature_extractor=VGG_model.predict(x_train)

features = feature_extractor.reshape(feature_extractor.shape[0], -1)

#This is our X input to RF
X_for_RF = features 

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 200, random_state = 0)

# Train the model on training data
# For sklearn no one hot encoding
RF_model.fit(X_for_RF, y_train) 

#Send test data through same feature extractor process
X_test_feature = VGG_model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

#Now predict using the trained RF model. 
prediction_RF = RF_model.predict(X_test_features)
#Inverse le transform to get original label back. 
prediction_RF = le.inverse_transform(prediction_RF)

#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction_RF) *100 ,"%" )
            ########################### Accuracy #######################
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, prediction_RF)
#print(cm)
sns.heatmap(cm, annot=True)


#Check results on a few select images

n=np.random.randint(0, x_test.shape[0])
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_feature=VGG_model.predict(input_img)
input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
prediction_RF = RF_model.predict(input_img_features)[0] 
prediction_RF = le.inverse_transform([prediction_RF])  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction_RF)
print("The actual label for this image is: ", test_labels[n])

#########################################8###########################

#### GUI #######


from tkinter import *
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk

root = tk.Tk()
root.geometry("1300x850")  # Size of the window 
#root.resizable(width=False, height=False)
root.title('Object Detector')
root['background']='#617884' 
font1=('times', 20, 'bold')
font2=('times', 15, 'bold')


label = tk.Label(root,text='Upload images & Detect',width=40,font=font1)
label['background']='#617884' 
label.grid(row=1,column=1)
label.place(anchor = CENTER, relx = .5, rely = .03)


upload= tk.Button(root, text='Click here to Upload Images',
                  width=30,command = lambda:upload_file() ,font=font2)
upload.grid(row=3,column=1,pady=5)
upload.place(anchor = CENTER, relx = .5, rely = .085)


def upload_file():
    f_types = [('Jpg Files', '*.jpg')]   # types of files to select 
    filename = tk.filedialog.askopenfilename(multiple=True,filetypes=f_types)
    col=1 # start from column 1
    row=3 # start from row 3 
    
    
    for pathgui in filename:
        img=Image.open(pathgui)# read the image file
        list_of_images = []
        img_preprocessed = cv2.imread(pathgui, cv2.IMREAD_COLOR)
        img_preprocessed = cv2.resize(img_preprocessed, (256,256))
        img_preprocessed = cv2.cvtColor(img_preprocessed, cv2.COLOR_RGB2BGR)
        list_of_images.append(img_preprocessed)
        arr = np.array(list_of_images)

        feature_extractor_input = VGG_model.predict(arr)
        features_input = feature_extractor_input.reshape(feature_extractor_input.shape[0], -1)
        prediction_input = RF_model.predict(features_input)[0] #edited
        prediction_input_Normal = le.inverse_transform([prediction_input]) #edited
        
        
        img=img.resize((144,144)) # new width & height
        img=ImageTk.PhotoImage(img)
        e1 =tk.Label(root)
        e1.grid(row=row,column=col,pady=100,padx=10)
        e1.image = img
        text_answer=prediction_input_Normal[0] #edited
        l2 = tk.Label(root,text=text_answer,width=20,font=font2)  
        l2.grid(row=row+1,column=col,pady=0,padx=10)
        e1['image']=img # garbage collection
        if(col==5): # start new line after third column
            row=row+2# start wtih next row
            col=1    # start with first column
        else:       # within the same row 
            col=col+1 # increase to next column                                                                                 

root.mainloop()














