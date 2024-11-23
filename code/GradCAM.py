
## This code ie developed using the official GradCam documentations
#https://keras.io/examples/vision/grad_cam/

##pip install tensorflow==2.15

import tensorflow as tf
print(tf.__version__)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from skimage import io
import glob
from sklearn.model_selection import train_test_split
tf.random.set_seed(21)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#from tensorflow.keras.callbacks import TensorBoard
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import *
from tensorflow.compat.v1.keras.layers import BatchNormalization
from tensorflow.compat.v1.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.compat.v1.keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from tensorflow.compat.v1.keras.optimizers import Adadelta, Adam
from tensorflow.compat.v1.keras.callbacks import ModelCheckpoint
from tensorflow import keras
keras.utils.set_random_seed(21)
from sklearn.preprocessing import MinMaxScaler

import pickle
with open('./WFBBDEM/WFBBDEM+/NEWFBB_DEM/NEWFBB_DEM/data','rb') as fin: data = pickle.load(fin)
with open('./WFBBDEM/WFBBDEM+/NEWFBB_DEM/NEWFBB_DEM/label','rb') as fin: label = pickle.load(fin)
with open('./WFBBDEM/WFBBDEM+/NEWFBB_DEM/NEWFBB_DEM/name','rb') as fin: name = pickle.load(fin)

#data=np.expand_dims(data,3) # add channel dimension, 4D
data = np.repeat(data[..., np.newaxis], 3, -1)

labelN=label.shape[0]
print(name)

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

# Last convolutional layer name
last_conv_layer_name = 'top_activation'

# Compute Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):

    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )


    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]


    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))


    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)


    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

    return heatmap

def save_and_display_gradcam(img, heatmap, test_data, test_label, pred_label, i, ax, alpha=0.8):

    img = np.uint8(255 * img)
    heatmap = np.uint8(255 * heatmap)
    jet = mpl.colormaps["jet"]

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Resize the heatmap to the same size as the image
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))  # Resize heatmap to image size
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on the original image with alpha
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Overlay the heatmap on the original image
    img = (test_data[i] * 255).astype('uint8')
    output = superimposed_img

    # Extract labels
    actual = test_label[i][0]  # Assuming test_label contains (actual, name)
    name = test_label[i][1]
    prediction = pred_label[i]  # Assuming pred_label contains predictions
    title = f'Actual: {actual}\nPrediction: {prediction}\nName: {name}'
    color = 'green' if prediction == actual else 'red'

    # Display original image and Grad-CAM
    ax.imshow(np.hstack([img, np.array(output)]))  # Display the image and heatmap side-by-side
    ax.axis('off')
    ax.set_title(title, fontsize=10, color=color)

# Load the model (assuming EfficientNetV2S)
model_builder = tf.keras.applications.EfficientNetV2S
model = model_builder(weights="./WFBBDEM/WFBBDEM+/NEWFBB_DEM/HRDEMDrop0.5EfficientNetS.h5",
                      input_shape=(100, 100, 3),
                      include_top=True,
                      pooling="max",
                      classes=2,
                      classifier_activation="softmax")

# Remove last layer's softmax for Grad-CAM
model.layers[-1].activation = None

# Predict
pred_label = model.predict(test_data)
print(pred_label)
pred_label = np.argmax(pred_label, axis=1)


for t in range(pred_label.shape[0]):
    NameWrong=[]
    labelPred=[]
    if (pred_label[t]!=test_label[t,0]):
        # print (test_label[t,0], test_label[t,1], pred_label[t])
        NameWrong.append(test_label[t,1])
        labelPred.append(pred_label[t])

"""##Single Image visualization"""

image_index = 13
img_array = np.expand_dims(test_data[image_index], axis=0)

# Get predictions and generate Grad-CAM heatmap
preds = model.predict(img_array)
pred_index = np.argmax(preds[0])  # Dynamically select the top predicted class
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=pred_index)

# Normalize the heatmap to [0, 1] range for accurate colormap application
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1  # Normalize to [0, 1]

# Convert the original image for display
img = (test_data[image_index] * 255).astype('uint8')  # Convert image to uint8 format

# Apply the jet color map to the normalized heatmap
jet = mpl.colormaps["jet"]  # Initialize the jet color map
colored_heatmap = jet(heatmap)[:, :, :3]
colored_heatmap = np.uint8(255 * colored_heatmap)  # Scale to [0, 255] for RGB image

# Resize the colored heatmap to match the original image size
colored_heatmap = keras.utils.array_to_img(colored_heatmap)
colored_heatmap = colored_heatmap.resize((img.shape[1], img.shape[0]))  # Resize to match original image
colored_heatmap = keras.utils.img_to_array(colored_heatmap)

# Overlay the colored heatmap on the original image with alpha blending
alpha = 0.8  # Set blending factor for the heatmap
superimposed_img = colored_heatmap * alpha + img  # Apply heatmap with transparency
superimposed_img = keras.utils.array_to_img(superimposed_img)  # Convert to image for display

# Display the result with color bar
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(superimposed_img)  # Show the superimposed image
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=ax, orientation="vertical", fraction=0.046, pad=0.04)  # Add color bar
cbar.set_label('Heatmap Intensity')  # Label the color bar

# Hide axes for a clean display
ax.axis('off')
plt.show()

"""##Multiple Image visualization"""

# Configuration
start_index =218       # Starting index for the images you want to process
num_images = 6          # Number of images to select and process
r, c = 2, 3             # Rows and columns for grid visualization (should match num_images)

# Prepare figure for the selected images
fig, axs = plt.subplots(r, c, figsize=(c * 8, r * 4))

# Loop over the images and generate Grad-CAM heatmaps
for i in range(start_index, start_index + num_images):
    img_array = np.expand_dims(test_data[i], axis=0)

    # Get predictions and generate Grad-CAM
    preds = model.predict(img_array)
    pred_index = np.argmax(preds[0])  # Dynamically select the top predicted class
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=pred_index)

    # Overlay the heatmap on the original image using the new function
    save_and_display_gradcam(
        test_data[i], heatmap, test_data, test_label, pred_label, i,
        axs[(i - start_index) // c, (i - start_index) % c], alpha=0.8
    )

plt.tight_layout()
plt.show()



## Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs except errors

# Array to store indices of misclassified instances
misclassified_indices = []

#Function to display a batch of misclassified instances
def display_misclassified_batch(batch_start, batch_size=6):
    r, c = 2, 3  # Set the grid layout
    fig, axs = plt.subplots(r, c, figsize=(c * 8, r * 4))

    # Loop over the batch and generate Grad-CAM visualizations
    for idx, mis_idx in enumerate(misclassified_indices[batch_start:batch_start + batch_size]):
        img_array = np.expand_dims(test_data[mis_idx], axis=0)
        preds = model.predict(img_array, verbose=0)  # Set verbose=0 to prevent output
        pred_index = np.argmax(preds[0])

        # Generate Grad-CAM heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=pred_index)

        # Display using save_and_display_gradcam function, directly passing pred_index
        save_and_display_gradcam(
            test_data[mis_idx], heatmap, test_data, test_label, pred_index, mis_idx,
            axs[idx // c, idx % c], alpha=0.8
        )

    plt.tight_layout()
    plt.show()


def save_and_display_gradcam(img, heatmap, test_data, test_label, pred_index, i, ax, alpha=0.8):
    # Convert image to uint8
    img = np.uint8(255 * img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Extract RGB values from colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Resize the heatmap to the same size as the image
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))  # Resize heatmap to image size
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on the original image with alpha
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Overlay the heatmap on the original image
    img = (test_data[i] * 255).astype('uint8')
    output = superimposed_img

    # Extract labels
    actual = test_label[i][0]  # Assuming test_label contains (actual, name)
    name = test_label[i][1]
    prediction = pred_index  # Use pred_index directly for prediction
    title = f'Actual: {actual}\nPrediction: {prediction}\nName: {name}'
    color = 'green' if prediction == actual else 'red'

    # Display original image and Grad-CAM
    ax.imshow(np.hstack([img, np.array(output)]))
    ax.axis('off')
    ax.set_title(title, fontsize=10, color=color)



"""##Images With Varying Spatial Locations"""


with open('./WFBBDEM/rotate/data1','rb') as fin: data2 = pickle.load(fin)
with open('.WFBBDEM/rotate/label1','rb') as fin: label2 = pickle.load(fin)
with open('./WFBBDEM/rotate/name1','rb') as fin: name2 = pickle.load(fin)

#data=np.expand_dims(data,3) # add channel dimension, 4D
data1 = np.repeat(data2[..., np.newaxis], 3, -1)

labelN=label2.shape[0]
print(name2)

label_name=np.empty((labelN,2),dtype=int) # label + name
for i in range(labelN):
    label_name[i,0]=label2[i] # label
    if (name2[i].isdigit()):
        name2[i]=name2[i]
    else:
        name2[i]=name2[i][1:] # only keep number, delete first letter
    name2[i] = name2[i].split()[0]
    label_name[i,1]=name2[i]

# Predict
pred_label = model.predict(test_data)
print(pred_label)
pred_label = np.argmax(pred_label, axis=1)


for t in range(pred_label.shape[0]):
    NameWrong=[]
    labelPred=[]
    if (pred_label[t]!=test_label[t,0]):
        # print (test_label[t,0], test_label[t,1], pred_label[t])
        NameWrong.append(test_label[t,1])
        labelPred.append(pred_label[t])


        

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs except errors

# Array to store indices of misclassified instances
misclassified_indices = []

# Loop over the entire dataset to identify misclassified instances
for i in range(len(test_data)):
    img_array = np.expand_dims(test_data[i], axis=0)
    preds = model.predict(img_array, verbose=0)  # Set verbose=0 to prevent output
    pred_index = np.argmax(preds[0])
    actual_label = test_label[i][0]
    predicted_label = pred_index

    # Check for misclassification
    if predicted_label != actual_label:
        misclassified_indices.append(i)

# Print the total number of misclassified instances
print(f"Total misclassified instances: {len(misclassified_indices)}")


def display_misclassified_batch(batch_start, batch_size=6):
    r, c = 2, 3  # Set the grid layout
    fig, axs = plt.subplots(r, c, figsize=(c * 8, r * 4))

    # Loop over the batch and generate Grad-CAM visualizations
    for idx, mis_idx in enumerate(misclassified_indices[batch_start:batch_start + batch_size]):
        img_array = np.expand_dims(test_data[mis_idx], axis=0)
        preds = model.predict(img_array, verbose=0)  # Set verbose=0 to prevent output
        pred_index = np.argmax(preds[0])

        # Generate Grad-CAM heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=pred_index)

        # Display using save_and_display_gradcam function
        save_and_display_gradcam(
            test_data[mis_idx], heatmap, test_data, test_label, pred_index, mis_idx,
            axs[idx // c, idx % c], alpha=0.8
        )

    plt.tight_layout()
    plt.show()

# Display the first batch of 6 misclassified instances
display_misclassified_batch(6)





