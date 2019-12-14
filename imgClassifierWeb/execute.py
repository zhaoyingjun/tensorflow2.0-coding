# coding=utf-8
#导入所有的依赖包
import  tensorflow as tf
import numpy as np
from cnnModel import cnnModel
import os
import pickle
import getConfig
import sys

gConfig = {}

gConfig=getConfig.get_config(config_file="config.ini")

def read_data(dataset_path, im_dim, num_channels,num_files,images_per_file):
       
        files_names = os.listdir(dataset_path)
        
     
        dataset_array = np.zeros(shape=(num_files * images_per_file, im_dim, im_dim, num_channels))
       
        dataset_labels = np.zeros(shape=(num_files * images_per_file), dtype=np.uint8)
        index = 0
        
        for file_name in files_names:
            if file_name[0:len(file_name)-1] == "data_batch_":
                print("正在处理数据 : ", file_name)
                data_dict = unpickle_patch(dataset_path + file_name)
                images_data = data_dict[b"data"]
                print(images_data.shape)
               
                images_data_reshaped = np.reshape(images_data, newshape=(len(images_data), im_dim, im_dim, num_channels))
                
                dataset_array[index * images_per_file:(index + 1) * images_per_file, :, :, :] = images_data_reshaped
                
                dataset_labels[index * images_per_file:(index + 1) * images_per_file] = data_dict[b"labels"]
                index = index + 1
        return dataset_array, dataset_labels  # 返回数据

def unpickle_patch(file):
    
    patch_bin_file = open(file, 'rb')
    patch_dict = pickle.load(patch_bin_file, encoding='bytes')
    return patch_dict

def create_model():
    
    if 'pretrained_model'in gConfig:
        model=tf.keras.models.load_model(gConfig['pretrained_model'])
        return model
    ckpt=tf.io.gfile.listdir(gConfig['working_directory'])

    
    if  ckpt:
        
        model_file=os.path.join(gConfig['working_directory'], ckpt[-1])
        print("Reading model parameters from %s" % model_file)
       

        model=tf.keras.models.load_model(model_file)
        return model
    else:
        model=cnnModel(gConfig['keeps'])
        model=model.createModel()
        return model


dataset_array, dataset_labels = read_data(dataset_path=gConfig['dataset_path'], im_dim=gConfig['im_dim'],
   num_channels=gConfig['num_channels'],num_files=5,images_per_file=gConfig['images_per_file'])

test_array, test_labels = read_data(dataset_path=gConfig['test_path'], im_dim=gConfig['im_dim'],
   num_channels=gConfig['num_channels'],num_files=1,images_per_file=gConfig['images_per_file'])

dataset_array=dataset_array.astype('float32')/255
test_array=test_array.astype('float32')/255

dataset_labels=tf.keras.utils.to_categorical(dataset_labels,10)
test_labels=tf.keras.utils.to_categorical(test_labels,10)

def train():
    
     model=create_model()
     print(model.summary())
    
     model.fit(dataset_array,dataset_labels,verbose=1,epochs=gConfig['epochs'],validation_data=(test_array,test_labels))

    
     filename='cnn_model.h5'
     checkpoint_path = os.path.join(gConfig['working_directory'], filename)
     model.save(checkpoint_path)
     sys.stdout.flush()

def predict(data):
    
    file = gConfig['dataset_path'] + "batches.meta"
    patch_bin_file = open(file, 'rb')
    
    label_names_dict = pickle.load(patch_bin_file)["label_names"]
    
    model=create_model()
    
    predicton=model.predict(data)
    
    index=tf.math.argmax(predicton[0]).numpy()
    
    return label_names_dict[index]

if __name__=='__main__':
    
    if gConfig['mode']=='train':
        train()
   
    elif gConfig['mode']=='serve':
        print('请使用:python3 app.py')

    
