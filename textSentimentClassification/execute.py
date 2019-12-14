
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import getConfig
import tensorflow.keras.preprocessing.sequence as sequence
import textClassiferModel as model
import time
UNK_ID=3

gConfig={}
gConfig=getConfig.get_config(config_file='config.ini')

sentence_size=gConfig['sentence_size']
embedding_size = gConfig['embedding_size']
vocab_size=gConfig['vocabulary_size']
model_dir = gConfig['model_dir']

def read_npz(data_file):
    r = np.load(data_file)
    return r['arr_0'],r['arr_1'],r['arr_2'],r['arr_3']


def pad_sequences(inp):
    out_sequences=sequence.pad_sequences(inp, maxlen=gConfig['sentence_size'],padding='post',value=0)
    return out_sequences
x_train, y_train, x_test, y_test =read_npz(gConfig['npz_data'])
x_train = pad_sequences(x_train)
x_test = pad_sequences(x_test)

dataset_train = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(gConfig['shuffle_size'])

dataset_test = tf.data.Dataset.from_tensor_slices((x_test,y_test)).shuffle(gConfig['shuffle_size'])

checkpoint_path = gConfig['model_dir']

ckpt_manager = tf.train.CheckpointManager(model.ckpt, checkpoint_path, max_to_keep=5)

def create_model():
    ckpt = tf.io.gfile.listdir(checkpoint_path)
    if ckpt:
        print("reload pretrained model")
        model.ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
        return model
    else:
        return model

def train():
    model=create_model()
    while True:
        model.train_loss.reset_states()
        model.train_accuracy.reset_states()

        for (batch,(inp, target)) in enumerate(dataset_train.batch(gConfig['batch_size'])):
            start = time.time()
            loss = model.step(inp,target)
            print ('训练集:Epoch {:} ,Batch {:} ,Loss {:.4f},Prestep {:.4f}'.format(epoch + 1, batch,loss.numpy(),(time.time()-start)))

        #for (batch,(inp,target)) in enumerate(dataset_test.batch(gConfig['batch_size'])):
            #start = time.time()
           # loss = model.evaluate(inp,target)
           # print ('测试集:Epoch {:} ,Batch {:} ,Loss {:.4f},Prestep {:.4f}'.format(epoch + 1, batch,loss.numpy(),(time.time()-start)))

        ckpt_save_path=ckpt_manager.save()
        print ('保存epoch{}模型在 {}'.format(epoch+1, ckpt_save_path))

def text_to_vector(inp):
    vocabulary_file=gConfig['vocabulary_file']
    tmp_vocab = []
    with open(vocabulary_file, "r") as f:#读取字典文件的数据，生成一个dict，也就是键值对的字典
         tmp_vocab.extend(f.readlines())
    tmp_vocab = [line.strip() for line in tmp_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])
    print(vocab)
    line_vec = []
    for words in inp.split():
        line_vec.append(vocab.get(words, UNK_ID))  #
    return line_vec

def predict(sentences):
    state=['pos','neg']
    model=create_model()
    indexes = text_to_vector(sentences)
    print(indexes)
    inp = pad_sequences([indexes])
    inp=tf.reshape(inp[0],(1,len(inp[0])))
    predictions=model.step(inp,inp,False)
    pred = tf.math.argmax(predictions[0])
    p=np.int32(pred.numpy())
    return state[p]

if __name__ == "__main__":
    if gConfig['mode']=='train':
        train()
    elif gConfig['mode']=='serve':
        print('Sever Usage:python3 app.py')



