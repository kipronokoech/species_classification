#!/usr/bin/env python
# coding: utf-8

# In[17]:


import sys
sys.path.insert(0,"/mnt/lustre3p/users/kkoech/.local")
#sys.path.append("./mrcnn")


# In[18]:


import os
os.chdir("./")


# In[19]:
import tensorflow as tf
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=10,inter_op_parallelism_threads=10)
sess = tf.compat.v1.Session(config=session_conf) 
# !pip uninstall -y tensorflow
# !pip uninstall -y tensorflow-gpu


# In[20]:


# !pip install --target="/mnt/lustre3p/users/kkoech/.local" tensorflow
# !pip install --target="/mnt/lustre3p/users/kkoech/.local" tensorflow-gpu


# In[21]:


# Set matplotlib backend so that figures are saved on background
import matplotlib
# matplotlib.use("Agg")
# get_ipython().run_line_magic('matplotlib', 'inline')

# Import necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout,Flatten,Input,Dense, BatchNormalization
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from matplotlib import pyplot as plt
import numpy as np
import cv2
import argparse
import pickle
from tensorflow.keras.applications import ResNet50, InceptionV3
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.models import load_model
import datetime
import random
import json
import collections
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set(font_scale=1.4)
import pandas as pd
from tqdm import tqdm
# get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[22]:



print(tf.__version__)
print(tf.__file__)


# In[7]:


# !pip3 install --target="/mnt/lustre3p/users/kkoech/.local" tensorflow-gpu


# In[8]:


dataset = "images/"
model = "models/"


# In[9]:


labels_list = [i for i in os.listdir(dataset)]
print(labels_list)
print(len(labels_list))


# In[10]:


all_details = []
for index, name in tqdm(enumerate(labels_list)):
    if os.path.exists("data_as_array/{}.npy".format(name)):
        continue
    print(name)
    files_in_folder = list(paths.list_images(os.path.join(dataset,name)))
    folder_data = []
    for index1, file1 in enumerate(files_in_folder):
        data_point = {}
        # extract label from image path
        label = file1.split(os.path.sep)[-2]
        # interested in only three classes, so, we skip the rest
        if label not in labels_list:
            continue
        # append the label to labels
        # labels.append([label])
        # load image, convert to RGB, resize to
        # 299-d 
        print(index1, file1)
        try:
            image = cv2.imread(file1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (299,299))
        except:
            print("error")
            continue
        # append image to image_data
        #image_data.append(image)
        data_point = {
            "image":image,
            "label": label
        }
        folder_data.append(data_point)
    category_data = np.array(folder_data)
    if not os.path.exists("data_as_array/{}.npy".format(name)):
        np.save("data_as_array/{}.npy".format(name),folder_data, allow_pickle=True)
    all_details.append(folder_data)


# In[11]:


data1 = []
for file1 in os.listdir("data_as_array"):
    data2 = np.load(os.path.join("data_as_array",file1), allow_pickle=True).tolist()
    data1.append(data2)
data1 = np.array(data1,dtype=object)


# In[12]:


num = random.choice(np.arange(0,len(data1)))
print(data1[num][0]["label"])
plt.imshow(data1[num][0]["image"]);
plt.grid(None)
plt.show()


# In[13]:


def train_val_test_split(data, subset_labels, display_proportions=True):
    # all labels
    all_labels = [i[0]["label"] for i in data if len(i)!=0]
    assert set(np.intersect1d(subset_labels, all_labels))==set(subset_labels),    "Wrong label given in subset_labels. Possible values: {}".format(all_labels)
    # encoder object
    global lb
    lb = LabelBinarizer()
    # Filter classes
    class_filtered = []
    train_data = []
    val_data = []
    test_data = []
    for index, elem in enumerate(data):
        if not elem:
            continue
        label = elem[0]["label"]
        if label not in subset_labels:
            continue
        train_class1, val_class1, test_class1 =               np.split(elem, 
                       [int(0.6*len(elem)), int(0.8*len(elem))])
        train_data.extend(list(train_class1))
        val_data.extend(list(val_class1))
        test_data.extend(list(test_class1))
        
    X_train, y_train = np.array([i["image"] for i in train_data]),    lb.fit_transform(np.array([j["label"] for j in train_data]))

    X_val, y_val = np.array([i["image"] for i in val_data]),     lb.fit_transform(np.array([j["label"] for j in val_data]))

    X_test, y_test = np.array([i["image"] for i in test_data]),    lb.fit_transform(np.array([j["label"] for j in test_data]))

    if display_proportions == True:
        train_dist = dict(collections.Counter([j["label"] for j in train_data]))
        print("For training", json.dumps(train_dist, indent=3))
        val_dist = dict(collections.Counter([j["label"] for j in val_data]))
        print("For validation", json.dumps(val_dist, indent=3))
        test_dist = dict(collections.Counter([j["label"] for j in test_data]))
        print("For testing", json.dumps(test_dist, indent=3))
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# In[14]:


# Because of possible computation limitations you can train the model(s)
# on the subset of data like we do here
subset_labels1 = ['antelope', 'baboon', 'bears', 'bird',                    'elephants', 'giraffe', 'hare', 'hippo',                   'leopard', 'lion', 'rhino', 'zebra']
(X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(data=data1,subset_labels=subset_labels1)


# In[ ]:


class SpeciesModel(object):
    def __init__(self, mode, model_path, logs_path, resume=False):
        assert mode in ["training", "inference"], "mode can be either training or inference string"
        self.mode = mode
        self.model_path = model_path
        self.logs_path = logs_path
        self.resume = resume
        if self.mode == "inference":
            self.Evaluation()

    def get_modePath(self):
        if self.resume == True:
            from pathlib import Path
            model_dir = sorted(Path(self.model_path).iterdir(), key=os.path.getmtime)[-1]
            print(model_dir)
            if not os.path.exists(model_dir) or os.path.isdir(model_dir)==False:
                model_dir = os.path.join(self.model_path,datetime.datetime.now().strftime("%m%d%Y-%H%M%S"))
                # We need to make this directory explicitly unline in logdir in Tensorboard callback
                os.makedirs(model_dir)
                return model_dir
            return model_dir
        else:
            model_dir = os.path.join(self.model_path,datetime.datetime.now().strftime("%m%d%Y-%H%M%S"))
            # We need to make this directory explicitly unline in logdir in Tensorboard callback
            os.makedirs(model_dir)
            return model_dir

    def get_logsPath(self):
        if self.resume==True:
            from pathlib import Path
            log_dir = sorted(Path(self.logs_path).iterdir(), key=os.path.getmtime)[-1]
            print(log_dir)
            return log_dir
        else:
            # tensorboard will make this logs_path automatically
            log_dir = os.path.join(self.logs_path,datetime.datetime.now().strftime("%m%d%Y-%H%M%S"))
            return log_dir

    def setCallbacks(self):
      
        # Model callbacks

        # Checkpoint - save the model at an epoch whenever there's an improvment in val_los
        # in relation to previous epoch
        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(self.get_modePath(),"model_{epoch:02d}-{val_loss:.2f}.h5"), monitor='val_loss',\
            verbose=1, save_best_only=True,save_weights_only=False, \
            mode='auto', save_freq='epoch'
        )

        # Stop training if the model is no longer improving
        early_stopping = EarlyStopping(
            monitor='val_loss', min_delta=0, patience=10, verbose=1,
            mode='auto', baseline=None, restore_best_weights=False
        )

        # Log the training stats into a tensorbord 
    
        tensorboard = TensorBoard(
            log_dir=self.get_logsPath(), histogram_freq=0, write_graph=True,
            write_images=False, update_freq='epoch', profile_batch=2,
            embeddings_freq=0, embeddings_metadata=None
        )

        # list of all the three call backs
        callbacks = [model_checkpoint, early_stopping, tensorboard]
        return callbacks

    def train(self, binarizer, epochs, lr=1e-4, resume_path=None):
        # initialize the training data augmentation object
        trainAug = ImageDataGenerator(
            rotation_range=30,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")
        # initialize the validation/testing data augmentation object (which
        # we'll be adding mean subtraction to)
        valAug = ImageDataGenerator()
        if self.resume == True:
            if resume_path  is None:
                from pathlib import Path
                assert len(os.listdir(self.model_path))>0,                "Training cannot be resumed because there's no saved model checkpoint. set resume to False"
                model_dir1 = sorted(Path(self.model_path).iterdir(), key=os.path.getmtime)[-1]
                assert os.path.isdir(model_dir1),                "Training cannot be resumed because there's no saved model checkpoint. set resume to False"
                    
                resume_path = sorted(Path(model_dir1).iterdir(), key=os.path.getmtime)[-1]
               
            loaded_model = load_model(resume_path)
            # retraining the model
            initial_epoch= int(str(resume_path).split("model_")[-1].split("-")[0])
            assert epochs>=initial_epoch,"epochs must be greater than or equals to initial epoch. Choice any value above {}".format(initial_epoch)
            # train the head of the network for a few epochs (all other layers
            # are frozen) -- this will allow the new FC layers to start to become
            # initialized with actual "learned" values versus pure random
            print("[INFO] training head...")
            H = loaded_model.fit(
                x=trainAug.flow(X_train, y_train, batch_size=32),
                validation_data=valAug.flow(X_val, y_val),
                epochs=epochs, callbacks=[self.setCallbacks()], initial_epoch=initial_epoch)

        elif self.resume == False:
            # load pre-trained model graph, don't add final layer i.e include_top=False
            base_model = InceptionV3(
                include_top=False,
                weights="imagenet",
                input_tensor=None,
                input_shape=(299,299,3),
            )

            # add global pooling just like in InceptionV3
            new_output = GlobalAveragePooling2D()(base_model.output)
            new_output = Dense(len(binarizer.classes_), activation='softmax')(new_output)
            model = Model(base_model.inputs, new_output)
            
            # set all layers trainable by default
            for layer in model.layers:
                layer.trainable = True
                if isinstance(layer, BatchNormalization):
                    # we do aggressive exponential smoothing of batch norm
                    # parameters to faster adjust to our new dataset
                    layer.momentum = 0.7

            # fix deep layers (fine-tuning only last 50)
            for layer in model.layers[:-50]:
                # fix all but batch norm layers, because we need to update moving averages for a new dataset!
                if not isinstance(layer, BatchNormalization):
                    layer.trainable = False
            # compile our model (this needs to be done after our setting our
            # layers to being non-trainable)
            print("[INFO] compiling model...")
            opt = SGD(learning_rate=lr, momentum=0.9, decay=1e-4/epochs)
            model.compile(loss="categorical_crossentropy", optimizer=opt,
                metrics=["accuracy"])
            

            # train the head of the network for a few epochs (all other layers
            # are frozen) -- this will allow the new FC layers to start to become
            # initialized with actual "learned" values versus pure random
            print("[INFO] traini4/1AY0e-g6M3r-Y3fNHUKOAg5T0SlL_jmwbnFDKhjOklIUz1loS6Y0liW2OHYQng head...")
            H = model.fit(
                x=trainAug.flow(X_train, y_train, batch_size=32),
                validation_data=valAug.flow(X_val, y_val),
                epochs=epochs, callbacks=[self.setCallbacks()], initial_epoch=0)

            return model

    def Evaluation(self,saved_model=None):
        print("[INFO] Confusion matrix:")
        if saved_model is None:
            # path to the saved model is not given get the path to the latest saved model.
            from pathlib import Path
            model_dir1 = sorted(Path(self.model_path).iterdir(), key=os.path.getmtime)[-1]
            assert os.path.isdir(model_dir1) and len(os.listdir(model_dir1))>0,            "No model saved in the following path {}".format(model_dir1)
                
            saved_model = sorted(Path(model_dir1).iterdir(), key=os.path.getmtime)[-1]

        loaded_model = load_model(saved_model)

        predictions = loaded_model.predict(x=X_test.astype("float32"), batch_size=32, verbose=1)

        y_pred = predictions.argmax(axis=1)
        y_truth = y_test.argmax(axis=1)

        cm = confusion_matrix(y_truth,y_pred)
        print(cm)

        df_cm = pd.DataFrame(cm, index = ["Actual-"+i for i in lb.classes_],
                  columns = ["Predict-"+i for i in lb.classes_])
        group_counts = ["{0:0.0f}".format(value) for value in
                        cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in
                                cm.flatten()/np.sum(cm)]

        labels = [f"{v1}\n{v2}" for v1, v2 in
                    zip(group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(len(subset_labels1),len(subset_labels1))

        plt.figure(figsize=(15,12))
        plt.xlabel("Predictions")
        plt.ylabel("Actual")
        sns.heatmap(df_cm,annot=labels,fmt='')
        plt.show()
    
        print("[INFO] Classification Report:")
        results_dict = classification_report(y_truth, y_pred,
                                             output_dict=True,
                                             target_names=lb.classes_)
        print(results_dict)
        return cm, results_dict

model_path = './models'
logs_path = "./logs"
lr_value = 1e-4
s = SpeciesModel(mode="training", 
                 model_path=model_path, 
                 logs_path=logs_path,
                 resume=True)
                
s.train(binarizer=lb,
        epochs=100,
        lr=lr_value)


# In[ ]:





# In[ ]:




