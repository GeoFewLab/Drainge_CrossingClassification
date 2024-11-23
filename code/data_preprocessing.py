# Data Preprocessing
# To simplify data management, preprocess all images and save them as separate pickle files.
# This approach ensures easy loading and reusability for future tasks.

import os
from skimage import io
import rasterio as rio
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pathDEM='./WFBBDEM/WFBBDEM+/DEM'   #DEM

# import rasterio as rio
def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)] # get F,T folder
    imgs=[]
    labels=[]
    imgs_name=[]
    for idx,folder in enumerate(cate): # idx-> 0:F; 1:T; folder-> F,T
        print(idx,folder)
        for im in glob.glob(folder+"/*.tif"):
            im = im.replace('\\','/')
#            print('reading the images:%s'%(im))

            img_name=os.path.basename(im)
            img_name=os.path.splitext(img_name)[0] # get file name

            #img=io.imread(im)
            with rio.open(im) as i:
              img = i.read()
            # Normalize the dataset-MaxMin
            img = np.squeeze(img)
            scaler = MinMaxScaler(feature_range=(0, 1))
            img = scaler.fit_transform(img)

            imgs.append(img)
            labels.append(idx)
            imgs_name.append(img_name) # image name

    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32), np.asarray(imgs_name)

data1,label1,name1=read_img(pathDEM)



# Save processed data
with open('./WFBBDEM/WFBBDEM+/NEWFBB_DEM/data','wb') as fin: pickle.dump(data1,fin)
with open('./WFBBDEM/WFBBDEM+/NEWFBB_DEM//label','wb') as fin: pickle.dump(label1,fin)
with open('./WFBBDEM/WFBBDEM+/NEWFBB_DEM/name','wb') as fin: pickle.dump(name1,fin)





















