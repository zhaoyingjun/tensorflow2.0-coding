import tensorflow as tf
import getConfig
gConfig={}
gConfig=getConfig.get_config()

class CycleGAN(object):
    def __init__(self,learning_rate,beta1,beta2):
        # 初始化输入数据的维度，分别是长、宽、通道数
        self.img_rows =gConfig['patch_dim']
        self.img_cols = gConfig['patch_dim']
        self.channels = gConfig['channels']
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        #初始化学习率和一阶、二阶估计参数
        self.learning_rate=learning_rate
        self.beta1=beta1
        self.beta2=beta2
        # 计算遮罩层
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # 初始化G和D的第一层的输出维度
        self.gf = 32
        self.df = 64

        # 初始化循环一致性函数的权重
        self.lambda_cycle = 10.0
        self.lambda_id = 0.1 * self.lambda_cycle

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate,self.beta1,self.beta2)
    def create_model(self):
        # 构建和编译识别器模型
        d_A = self.build_discriminator()
        d_B = self.build_discriminator()
        d_A.compile(loss='mse',optimizer=self.optimizer,metrics=['accuracy'])
        d_B.compile(loss='mse',optimizer=self.optimizer,metrics=['accuracy'])
        # 构建生成器模型
        g_AB = self.build_generator()
        g_BA = self.build_generator()

        # 定义A、B两个域的输入层
        img_A = tf.keras.Input(shape=self.img_shape)
        img_B = tf.keras.Input(shape=self.img_shape)
        # 分别使用生成器g_AB和g_BA转换A、B域的图像
        fake_B = g_AB(img_A)
        fake_A = g_BA(img_B)
        #将转换后的图像进行逆转换，还原到其原始的域中
        reconstr_A = g_BA(fake_B)
        reconstr_B = g_AB(fake_A)
        # Identity mapping of images
        img_A_id = g_BA(img_A)
        img_B_id = g_AB(img_B)

        # 在训练生成器时，识别器的参数是不更新的
        d_A.trainable = False
        d_B.trainable = False

        # 识别器识别生成器生成的图像真假
        valid_A = d_A(fake_A)
        valid_B = d_B(fake_B)

        #构建一个生成器训练模型，两个生成器模型g_AB和g_BA是共享的
        combined = tf.keras.Model(inputs=[img_A, img_B],outputs=[ valid_A, valid_B,reconstr_A, reconstr_B,img_A_id, img_B_id ])
        #编译生成器训练模型
        combined.compile(loss=['mse', 'mse','mae', 'mae','mae', 'mae'],
                         loss_weights=[  1, 1,self.lambda_cycle, self.lambda_cycle,self.lambda_id, self.lambda_id ],
                         optimizer=self.optimizer)
        #返回构建的生成器模型和识别器模型
        return g_AB,g_BA,d_A,d_B,combined
    #定义生成器构建函数
    def build_generator(self):
        #定义卷积层构建方法
        def conv2d(layer_input, filters, f_size=4):
            #一层二维卷积层
            d = tf.keras.layers.Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            #激活函数是LeakyReLU
            d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
            #使用LayerNormalization代替InstanceNorNalization
            d = tf.keras.layers.LayerNormalization()(d)
            return d
        #定义逆卷积构建方法
        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            #
            u = tf.keras.layers.UpSampling2D(size=2)(layer_input)
            u = tf.keras.layers.Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = tf.keras.layers.Dropout(dropout_rate)(u)
            # 使用LayerNormalization代替InstanceNorNalization
            u = tf.keras.layers.LayerNormalization()(u)
            u = tf.keras.layers.Concatenate()([u, skip_input])
            return u

        # 定义神经网络输入层
        d0 = tf.keras.Input(shape=self.img_shape)

        # 构建卷积神经网络，一共四层卷积
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        # 构建拟卷积神经网络，前三层是拟卷积，最后一层采用上采样网络
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)
        u4 = tf.keras.layers.UpSampling2D(size=2)(u3)
        #最后一层是一层卷积作为输出，生成图像数据
        output_img = tf.keras.layers.Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return tf.keras.Model(d0, output_img)
    #定义识别器构建函数
    def build_discriminator(self):
    #定义识别器网络层,包括一层卷积层和一层激活层
        def d_layer(layer_input, filters, f_size=4, normalization=True):
            d = tf.keras.layers.Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = tf.keras.layers.LayerNormalization()(d)
            return d
        #
        img = tf.keras.Input(shape=self.img_shape)
        #构建识别器模型的网络结构，一共是四层，每层的输出按照背书df的偶数倍增加
        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)
        #使用一层二维卷积来作为输出层
        validity = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        #返回构建的网络模型
        return tf.keras.Model(img, validity)




                



        




 
    
    
    
   
