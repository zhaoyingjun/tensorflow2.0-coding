# coding=utf-8
#导入所需要的依赖包
import tensorflow as tf
import getConfig
gConfig={}
gConfig=getConfig.get_config(config_file='config.ini')
#将学习率设置为1
tf.keras.backend.set_learning_phase(0)
#加载已经训练完成的模型
model = tf.keras.models.load_model(gConfig['model_file'])
print(model.summary())
#model = tf.keras.applications.MobileNet()
export_path = gConfig['exeport_dir']
#使用saved_mode.experimental.export_saved_model的save方法完成模型文件的导出
tf.keras.experimental.export_saved_model(model,export_path,
        #inputs={'input_image': model.input}
       # outputs={t.name: t for t in model.outputs}
       serving_only=True
)

print('模型导出完成，并保存在：',export_path)
loaded = tf.keras.experimental.load_from_saved_model(export_path)
print(list(loaded.signatures.keys()))
