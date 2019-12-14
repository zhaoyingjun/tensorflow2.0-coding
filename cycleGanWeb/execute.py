# coding=utf-8
import tensorflow as tf
import  os
import sys
import  numpy as np
import cycleganModel
from data_loader import DataLoader
import getConfig
gConfig={}
gConfig=getConfig.get_config()
img_rows =gConfig['patch_dim']
img_cols = gConfig['patch_dim']
channels = gConfig['channels']
img_shape = (img_rows, img_cols, channels)
patch = int(img_rows / 2**4)
disc_patch = (patch, patch, 1)
data_loader = DataLoader(dataset_name=gConfig['dataset_name'],img_res=(img_rows, img_cols))


g_AB_model_file = os.path.join(gConfig['model_data'],"g_AB.h5")
g_BA_model_file = os.path.join(gConfig['model_data'], "g_BA.h5")
d_A_model_file = os.path.join(gConfig['model_data'], "d_A.h5")
d_B_model_file = os.path.join(gConfig['model_data'], "d_B.h5")
comb_model_file = os.path.join(gConfig['model_data'], "comb.h5")
ckpt = tf.io.gfile.listdir(gConfig['model_data'])
def create_model():

    # 判断是否已经有Model文件存在，如果model文件存在则加载原来的model并在原来的moldel继续训练，如果不存在则新建model相关文件
    if  ckpt:

        print("Reading model parameters from %s" % g_AB_model_file)
        g_AB_model = tf.keras.models.load_model(g_AB_model_file)

        print("Reading model parameters from %s" % g_BA_model_file)
        g_BA_model = tf.keras.models.load_model(g_AB_model_file)


        print("Reading model parameters from %s" % d_A_model_file)
        d_A_model = tf.keras.models.load_model(d_A_model_file)


        print("Reading model parameters from %s" % d_B_model_file)
        d_B_model = tf.keras.models.load_model(d_B_model_file)


        print("Reading model parameters from %s" % comb_model_file)
        comb_model = tf.keras.models.load_model(comb_model_file)

        return g_AB_model,g_BA_model,d_A_model,d_B_model,comb_model
    else:
        model = cycleganModel.CycleGAN(gConfig['learning_rate'], gConfig['beta1'], gConfig['beta2'])
        return model.create_model()

def train():
    g_AB_model,g_BA_model,d_A_model,d_B_model,comb_model=create_model()
    while True:
        for i in range( gConfig['dis_epoches_pergen']):
           for batch_i,(imgs_A,imgs_B) in enumerate(data_loader.load_batch(gConfig['batch_size'])):
               # Adversarial loss ground truths
               valid = np.ones((gConfig['batch_size'],) + disc_patch)
               fake = np.zeros((gConfig['batch_size'],) + disc_patch)

               # Translate images to opposite domain

               fake_B = g_AB_model.predict(imgs_A, steps=1)
               fake_A = g_BA_model.predict(imgs_B, steps=1)

               # Train the discriminators (original images = real / translated = Fake)
               dA_loss_real =d_A_model.train_on_batch(imgs_A, valid)
               dA_loss_fake = d_A_model.train_on_batch(fake_A, fake)
               dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

               dB_loss_real = d_B_model.train_on_batch(imgs_B, valid)
               dB_loss_fake = d_B_model.train_on_batch(fake_B, fake)
               dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

               d_loss = 0.5 * np.add(dA_loss, dB_loss)
               print("识别器loss:",d_loss)

        for batch_i, (imgs_A, imgs_B) in enumerate(data_loader.load_batch(gConfig['batch_size'])):
            g_loss = comb_model.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B])
            print("生成器loss:",g_loss)

        tf.keras.models.save_model(g_BA_model,g_BA_model_file,save_format='h5')
        tf.keras.models.save_model(g_AB_model, g_AB_model_file,save_format='h5')
        tf.keras.models.save_model(d_A_model, d_A_model_file,save_format='h5')
        tf.keras.models.save_model(d_B_model, d_B_model_file,save_format='h5')
        tf.keras.models.save_model(comb_model, comb_model_file,save_format='h5')


def gen(img,gen_AB):
    g_AB_model,g_BA_model,_,_,_=create_model()

    if gen_AB:
        img_AB=g_AB_model.predict(img)
        return img_AB
    else:
        img_BA=g_BA_model.predict(img)
        return img_BA
if __name__=='__main__':
    if len(sys.argv) - 1:
        gConfig = getConfig(sys.argv[1])
    else:
        # get configuration from config.ini
        gConfig = getConfig.get_config()
    if gConfig['mode']=='train':
        train()
    elif gConfig['mode']=='server':
        print('Sever Usage:python3 app.py')



















# self.combined.save("Gen_model.h5")
# self.g_AB.save("g_AB.h5")
# self.g_BA.save("g_BA.h5")

# Plot the progress
#print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
 #     % (epoch, epochs,
  #       batch_i, self.data_loader.n_batches,
   #      d_loss[0], 100 * d_loss[1],
    #     g_loss[0],
     #    np.mean(g_loss[1:3]),
       #  np.mean(g_loss[3:5]),
        # np.mean(g_loss[5:6]),
        # elapsed_time))








# if batch_i % sample_interval == 0:
#     self.sample_images(epoch, batch_i)
#
#
# def sample_images(self, epoch, batch_i):
#     os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
#     r, c = 2, 3
#
#     # imgs_A = self.data_loader.load_data(domain="A", batch_size=1, is_testing=True)
#     # imgs_B = self.data_loader.load_data(domain="B", batch_size=1, is_testing=True)
#
#     # Demo (for GIF)
#     imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
#     imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')
#
#     # Translate images to the other domain
#     fake_B = self.g_AB.predict(imgs_A)
#     # print(fake_B)
#     fake_A = self.g_BA.predict(imgs_B)
#     # Translate back to original domain
#     reconstr_A = self.g_BA.predict(fake_B)
#     reconstr_B = self.g_AB.predict(fake_A)
#
#     gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])
#
#     # Rescale images 0 - 1
#     gen_imgs = 0.5 * gen_imgs + 0.5
