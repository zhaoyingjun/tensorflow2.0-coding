import getConfig
from flask import Flask,render_template,request,make_response,jsonify
from werkzeug.utils import secure_filename
import os
import execute
from datetime import timedelta
import numpy as np
from PIL import Image
import cv2

import matplotlib.pyplot as plt

gConfig={}

gConfig=getConfig.get_config(config_file='config.ini')


def load_img(path):
    img = np.asarray(Image.open(path).resize((128,128)), dtype=np.uint8)
    img = img / 127.5 - 1.
    return img[np.newaxis, :, :, :]

def trans(img_path,style):

 img=load_img(img_path)
 if int(style)==1:
   img=execute.gen(img,True)
   return img
 else:
   return execute.gen(img,False)


"""下面是一个APP应用，作用很简单就是将图片上传，并显示风格迁移后的图片"""

#设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)
 
@app.route('/', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        f = request.files['file']
 
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
 
        style_input = request.form.get("name")

        # 当前文件所在路径
        basepath = os.path.dirname(__file__)
        # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        upload_path = os.path.join(basepath, 'predict_data/Original',secure_filename(f.filename))
        translated_path = os.path.join(basepath, 'predict_data/Translated', secure_filename(f.filename))
        f.save(upload_path)
        img=load_img(upload_path)
        image_data=trans(upload_path,style_input)
        image_data =image_data*0.5+0.5
        gen_imgs = np.concatenate([img, image_data])
        r, c = 1,2
        titles = ['Original', 'Translated']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for j in range(c):
            axs[j].imshow(gen_imgs[cnt])
            axs[j].set_title(titles[j])
            axs[j].axis('off')
            cnt += 1
        fig.savefig(translated_path)
        img_data=open(translated_path, "rb").read()

        response = make_response(img_data)
        response.headers['Content-Type'] = 'image/png'
        return response
 
    return render_template('upload.html')

if __name__ == '__main__':
   app.run(host = '0.0.0.0',port = 8989,debug= False)

