"""
Classifying Drainage Crossings on High-Resolution Digital Elevation Models Using Deep Learning Models: EfficientNET

EfficientNet Model
------------------

EfficientNet is a family of convolutional neural networks (CNNs) designed for image classification.
Developed by Google AI, it achieves high accuracy while being computationally efficient through
a method called compound scaling, which uniformly scales depth, width, and resolution.

Architecture
------------

- Baseline Network: Starts with a simple architecture, often derived from MobileNetV2.
- Compound Scaling: Scales network dimensions (depth, width, resolution) using a compound coefficient.
- Building Blocks: Utilizes mobile inverted bottleneck MBConv blocks, which include depthwise separable convolutions.
- Swish Activation: Employs the Swish activation function, which improves performance over ReLU.

Variants
--------

EfficientNetV2 includes variants Large to Small. Each variant balances accuracy and computational efficiency.

Applications
------------

EfficientNet models are widely used in image classification tasks and are ideal for applications requiring
high accuracy with limited computational resources.

Figure: EfficientNet Architecture (Refer to the related documentation or online sources for details.)

"""

import tensorflow as tf
print(tf.__version__)
tf.compat.v1.reset_default_graph()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from skimage import io
import glob
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import BatchNormalization
from tensorflow.compat.v1.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.compat.v1.keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from tensorflow.compat.v1.keras.optimizers import Adadelta, Adam
from tensorflow.compat.v1.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()




with open('./cnn/hrdem/data1','rb') as fin: data = pickle.load(fin)
with open('./cnn/hrdem/label1','rb') as fin: label = pickle.load(fin)
with open('./cnn/hrdem/name1','rb') as fin: name = pickle.load(fin)

#data=np.expand_dims(data,3) # add channel dimension, 4D
data = np.repeat(data[..., np.newaxis], 3, -1)

labelN=label.shape[0]


label_name=np.empty((labelN,2),dtype=int) # label + name
for i in range(labelN):
    label_name[i,0]=label[i] # label
    if (name[i].isdigit()):
        name[i]=name[i]
    else:
        name[i]=name[i][1:] # only keep number, delete first letter
    name[i] = name[i].split()[0]
    label_name[i,1]=name[i]

# split data into train and test groups
train_data,test_data,train_label,test_label = train_test_split(data,label_name,test_size=0.2,stratify=label,random_state=21)


LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 100

model1 = tf.keras.applications.EfficientNetV2S(
    #weights = './WFBBDEM/Models/ModelDEM2022EfficientNetS1.h5',
    input_shape = (100,100,3),
    include_top = False,
    input_tensor = None,
    pooling = "max",
    classes = 2,
    classifier_activation = "softmax")

classification_head = keras.Sequential(
    [        keras.layers.Dense(2, activation='softmax')],
    name='classification_head')

output_a = classification_head(model1.output)
model1 = keras.Model(model1.inputs, [output_a])

for layer in model1.layers:
    if isinstance(layer, Dropout):
        layer.rate = 0.5

model = keras.models.clone_model(model1)

OUT_DIR = "./model/cnn/"
checkpoint = ModelCheckpoint(os.path.join(OUT_DIR, 'HRDEMDrop0.5EfficientNetS.h5'),  # model filename
                              monitor='val_loss', # quantity to monitor
                              verbose=1, # verbosity - 0 or 1
                              save_best_only=True, # The latest best model will not be overwritten
                              save_weights_only=True, # save model, not only weights
                              mode='auto') # The decision to overwrite model is made
                             # automatically depending on the quantity to monitor

callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=10, min_lr=0, verbose=1, min_delta=0.001),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0,min_delta=0.001),
        checkpoint]

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=LEARNING_RATE),
              metrics=['accuracy'])

model.set_weights(model1.get_weights())

model_details = model.fit(train_data, train_label[:,0],
                          batch_size = BATCH_SIZE,
                          epochs = EPOCHS,
                          validation_split=0.2,
                          callbacks=callbacks,
                          verbose=1)

#### plot confusion matrix
def plot_confusion_matrix(confusion_mat):
    plt.imshow(confusion_mat,interpolation='nearest',cmap=plt.cm.Wistia)
    plt.title('CNN Confusion Matrix',fontsize=20)
    plt.colorbar()
    tick_marks=np.arange(2) # class number
    plt.xticks(tick_marks,tick_marks,fontsize=16)
    plt.yticks(tick_marks,tick_marks,fontsize=16)
    plt.ylabel('True Label',fontsize=16)
    plt.xlabel('Predicted Label',fontsize=16)
    for i in range(len(confusion_mat)):    #row
        for j in range(len(confusion_mat[i])):    #col
            plt.text(j, i, confusion_mat[i][j],fontsize=16) # images number of each part
    plt.show()

# Predict
scroe, accuracy = model.evaluate(test_data, test_label[:,0], batch_size=32)
pred_label = model.predict(test_data)
#pred_label = [sum([i[0] for i in q]) for q in test_data]
pred_label = np.argmax(pred_label, axis=1)


for t in range(pred_label.shape[0]):
    NameWrong=[]
    labelPred=[]
    if (pred_label[t]!=test_label[t,0]):
        # print (test_label[t,0], test_label[t,1], pred_label[t])
        NameWrong.append(test_label[t,1])
        labelPred.append(pred_label[t])
confusion_matrix = tf.math.confusion_matrix(labels=test_label[:,0],predictions=pred_label, num_classes=2, dtype=tf.int32, name=None, weights=None)
sess=tf.compat.v1.Session()

# #with tf.compat.v1.Session(graph=g) as sess:
confusion_matrix = sess.run(confusion_matrix)
plot_confusion_matrix(confusion_matrix)

#plot_learning_curves(model_details)
precision = (confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0])+confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[0][1]))/2
recall = (confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1])+confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[1][0]))/2
F = 2*(precision*recall)/(precision+recall)
print ('Accuracy:', '{:.4f}'.format(accuracy), 'Loss:','{:.4f}'.format(scroe), f'Precision: {precision}', f'Recall: {recall}', f'F1: {F}')

temp = model_details.history.pop('lr')

model_details.history.keys()

######################### learning curve

def plot_learning_curves(history):
    df=pd.DataFrame(history.history,index=np.arange(0, len(model_details.history['accuracy'])))
    df.plot(use_index=True,figsize=(8, 5))
    plt.grid(True)

    #plt.gca().set_ylim(0, 1)
    plt.show()

plot_learning_curves(model_details)

print ('Accuracy:', '{:.4f}'.format(accuracy), 'Batch Size:', BATCH_SIZE,'Learning Rate:', LEARNING_RATE, 'Epochs:', EPOCHS)



