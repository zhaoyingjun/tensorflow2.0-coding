# coding=utf-8
#导入需要的依赖包
import tensorflow as tf
import getConfig

gConfig={}

gConfig=getConfig.get_config(config_file='config.ini')

class cnnModel(object):
 
    def __init__(self,rate):
        self.rate=rate
    def createModel(self):
       
        model = tf.keras.Sequential()
        
        model.add(tf.keras.layers.Conv2D(64, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                                    input_shape=(32,32,3),name="conv1"))
        model.add(tf.keras.layers.Conv2D(64, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                                    name="conv2"))
        
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='valid', name="pool1"))

        model.add(tf.keras.layers.Dropout(rate=self.rate, name="d1"))
     
        model.add(tf.keras.layers.BatchNormalization())
        
        model.add(tf.keras.layers.Conv2D(128, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                                   name="conv3"))
        model.add(tf.keras.layers.Conv2D(128, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                                    name="conv4"))
        
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='valid', name="pool2"))
  
        model.add(tf.keras.layers.Dropout(rate=self.rate, name="d2"))
       
        model.add(tf.keras.layers.BatchNormalization())
        
        model.add(tf.keras.layers.Conv2D(256, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                                   name="conv5"))
        model.add(tf.keras.layers.Conv2D(256, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                                   name="conv6"))
        
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='valid', name="pool3"))

        model.add(tf.keras.layers.Dropout(rate=self.rate, name="d3"))
        
        model.add(tf.keras.layers.BatchNormalization())
       
        model.add(tf.keras.layers.Flatten(name="flatten"))
       
        model.add(tf.keras.layers.Dropout(self.rate))
       
        model.add(tf.keras.layers.Dense(128, activation='relu',kernel_initializer='he_normal'))
       
        model.add(tf.keras.layers.Dropout(self.rate))
        
        model.add(tf.keras.layers.Dense(10, activation='softmax',kernel_initializer='he_normal'))
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
       
        return model




