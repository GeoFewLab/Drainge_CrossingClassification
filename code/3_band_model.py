
import tensorflow as tf
print(tf.__version__)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from skimage import io
import glob
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow.keras.callbacks as callbacks  # Use TensorFlow 2 callbacks
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.pyplot as plt

keras.utils.set_random_seed(21)
tf.config.experimental.enable_op_determinism()

OUT_DIR ='./WFBBDEM/WFBBDEM+/'



import pickle
with open('./WFBBDEM/TPI_21_DATA/data','rb') as fin: data1 = pickle.load(fin)
with open('./WFBBDEM/TPI_21_DATA/label','rb') as fin: label1 = pickle.load(fin)
with open('./WFBBDEM/TPI_21_DATA/name','rb') as fin: name1 = pickle.load(fin)

with open('./WFBBDEM/pos/data','rb') as fin: data4 = pickle.load(fin)
with open('./WFBBDEM/pos/label','rb') as fin: label4 = pickle.load(fin)
with open('./WFBBDEM/pos/name','rb') as fin: name4 = pickle.load(fin)

with open('./WFBBDEM/NEWFBB_DEM/data','rb') as fin: data5 = pickle.load(fin)
with open('./WFBBDEM/NEWFBB_DEM/label','rb') as fin: label5 = pickle.load(fin)
with open('./WFBBDEM/NEWFBB_DEM/name','rb') as fin: name5 = pickle.load(fin)

datas = [[data1,label1,name1],[data4,label4,name4],[data5,label5,name5]]



#data=np.expand_dims(data,3) # add channel dimension, 4D
temp = []
for data,label,name in datas:
  #data = np.repeat(data[..., np.newaxis], 3, -1)

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
  temp.append([data,label,name,label_name])
datas = temp

data = []
for i in datas:
  temp = list(sorted(zip(*map(list,i)),key=lambda x:x[2]))
  data.append(temp)
datas = data

tt = list(zip(*datas))
temp = [np.asarray([i[0] for i in q]) for q in tt]
data = np.asarray([[[q[:,x,y] for x in range(100)] for y in range(100)] for q in temp])
label_name = [q[0][3] for q in tt]



train_data,test_data,train_label,test_label = train_test_split(data,label_name,test_size=0.2,stratify=label,random_state=21)

LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 100


model1 = tf.keras.applications.EfficientNetV2S(
    input_shape = (100,100,3),
    include_top = False,
    input_tensor = None,
    pooling = "max",
    classes = 2,
    classifier_activation = "softmax"
)

classification_head = keras.Sequential(
    [
        keras.layers.Dense(2, activation='softmax')
    ],
    name='classification_head'
)

output_a = classification_head(model1.output)
model1 = keras.Model(model1.inputs, [output_a])

for layer in model1.layers:
    if isinstance(layer, Dropout):
        layer.rate = 0.5

model = keras.models.clone_model(model1)



checkpoint = ModelCheckpoint(os.path.join(OUT_DIR, '3band.weights.h5'),  # model filename
                              monitor='val_accuracy', # quantity to monitor
                              verbose=1, # verbosity - 0 or 1
                              save_best_only= True, # The latest best model will not be overwritten
                              save_weights_only=True, # save model, not only weights
                              mode='auto') # The decision to overwrite model is made
                                            # automatically depending on the quantity to monitor

callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=10, min_lr=0, verbose=1,),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0),
        checkpoint
              ]

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=LEARNING_RATE),
              metrics=['accuracy'])

model.set_weights(model1.get_weights())

model_details = model.fit(np.asarray(train_data), np.asarray(train_label)[:,0],
                          batch_size = BATCH_SIZE,
                          epochs = EPOCHS,
                          validation_split=0.2,
                          callbacks=callbacks,
                          verbose=1)

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

def plot_learning_curves(history):
    df = pd.DataFrame(history.history, index=np.arange(0, len(history.history['loss'])).astype(str))
    df.plot(use_index=True, figsize=(8, 5))
    plt.grid(True)

test_label = np.asarray(test_label)

plot_learning_curves(model_details)

loss, accuracy = model.evaluate(test_data, test_label[:, 0], batch_size=BATCH_SIZE)

predictions = model.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)

wrong_names = []
predicted_wrong_labels = []
for i in range(predicted_labels.shape[0]):
  if predicted_labels[i] != test_label[i, 0]:
    wrong_names.append(test_label[i, 1])
    predicted_wrong_labels.append(predicted_labels[i])

confusion_matrix = tf.math.confusion_matrix(labels=test_label[:, 0], predictions=predicted_labels, num_classes=2, dtype=tf.int32)

confusion_matrix = confusion_matrix.numpy()

plot_confusion_matrix(confusion_matrix)

precision = (confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0]) +
             confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1])) / 2
recall = (confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1]) +
           confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0])) / 2
F = 2 * (precision * recall) / (precision + recall)

print('Accuracy:', '{:.4f}'.format(accuracy), 'Loss:', '{:.4f}'.format(loss),
      f'Precision: {precision}', f'Recall: {recall}', f'F1: {F}',
      'Batch Size:', BATCH_SIZE, 'Learning Rate:', LEARNING_RATE, 'Epochs:', EPOCHS)