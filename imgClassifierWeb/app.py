# coding=utf-8

import flask
import werkzeug
import os
import getConfig
import numpy as np
import execute
from PIL import Image

gConfig = {}

gConfig = getConfig.get_config(config_file='config.ini')


app = flask.Flask("imgClassifierWeb")


def CNN_predict():
        global secure_filename
        
        img = Image.open(os.path.join(app.root_path, 'predict_img/'+secure_filename))
      
        img = img.convert("RGB")
        
        r, g, b = img.split()
        r_arr = np.array(r)
        g_arr = np.array(g)
        b_arr = np.array(b)
        img = np.concatenate((r_arr, g_arr, b_arr))
      
        image = img.reshape([1, 32, 32, 3])/255
      
        predicted_class = execute.predict(image)
        print(predicted_class)
     
        return flask.render_template(template_name_or_list="prediction_result.html",
                                 predicted_class=predicted_class)



app.add_url_rule(rule="/predict/", endpoint="predict", view_func=CNN_predict)


def upload_image():
    global secure_filename
    if flask.request.method == "POST":  
        img_file = flask.request.files["image_file"]  
        secure_filename = werkzeug.secure_filename(img_file.filename)  # 生成一个没有乱码的文件名
        img_path = os.path.join(app.root_path, "predict_img/"+secure_filename)  # 获取图片的保存路径
        img_file.save(img_path)  
        print("图片上传成功.")

        return flask.redirect(flask.url_for(endpoint="predict"))
    return "图片上传失败"


app.add_url_rule(rule="/upload/", endpoint="upload", view_func=upload_image, methods=["POST"])

def predirect_upload():
    return flask.render_template(template_name_or_list="upload_image.html")

"""
"""
app.add_url_rule(rule="/", endpoint="homepage", view_func=predirect_upload)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8008, debug=False)
