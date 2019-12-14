#导入所有依赖包
import flask
import werkzeug
import os
import getConfig
import numpy as np
import execute
from PIL import Image
#初始化一个字典，用于存放从配置文件中获取的配置参数
gConfig = {}
#使用get_config方法从配置文件中获取配置参数
gConfig = getConfig.get_config(config_file='config.ini')

#创建一个flask wen应用，名称为imgClassifierWeb
app = flask.Flask("imgClassifierWeb")

#定义一个函数，初步处理获得数据并调用execute中的方法进行预存
def CNN_predict():
        global secure_filename
        #使用PIL中 的Image打开文件并获取图像文件中的信息
        img = Image.open(os.path.join(app.root_path, 'predict_img/'+secure_filename))
        #将图像文件的格式转换为RGB
        img = img.convert("RGB")
        #分别获取r,g,b三元组的像素数据并进行拼接
        r, g, b = img.split()
        r_arr = np.array(r)
        g_arr = np.array(g)
        b_arr = np.array(b)
        img = np.concatenate((r_arr, g_arr, b_arr))
        #将拼接得到的数据按照模型输入维度需要转换为（32，32，3)，并对数据进行归一化
        image = img.reshape([1, 32, 32, 3])/255
        #调用execute中的predict方法进行预测
        predicted_class = execute.predict(image)
        print(predicted_class)
        #将预测结果返回并使用模板进行页面渲染
        return flask.render_template(template_name_or_list="prediction_result.html",
                                 predicted_class=predicted_class)


"""
flask路由系统：
1、使用flask.Flask.route() 修饰器。
2、使用flask.Flask.add_url_rule()函数。
3、直接访问基于werkzeug路由系统的flask.Flask.url_map.
参考知识链接：https://www.jianshu.com/p/e69016bd8f08
1、@app.route('/index.html')
    def index():
        return "Hello World!"
2、def index():
    return "Hello World!"
    index = app.route('/index.html')(index)
app.add_url_rule:app.add_url_rule(rule,endpoint,view_func)
关于rule、ednpoint、view_func以及函数注册路由的原理可以参考：https://www.cnblogs.com/eric-nirnava/p/endpoint.html
"""
app.add_url_rule(rule="/predict/", endpoint="predict", view_func=CNN_predict)
"""
知识点：
flask.request属性
form: 
一个从POST和PUT请求解析的 MultiDict（一键多值字典）。
args: 
MultiDict，要操作 URL （如 ?key=value ）中提交的参数可以使用 args 属性:
searchword = request.args.get('key', '')
values: 
CombinedMultiDict，内容是form和args。 
可以使用values替代form和args。
cookies: 
顾名思义，请求的cookies，类型是dict。
stream: 
在可知的mimetype下，如果进来的表单数据无法解码，会没有任何改动的保存到这个·stream·以供使用。很多时候，当请求的数据转换为string时，使用data是最好的方式。这个stream只返回数据一次。
headers: 
请求头，字典类型。
data: 
包含了请求的数据，并转换为字符串，除非是一个Flask无法处理的mimetype。
files: 
MultiDict，带有通过POST或PUT请求上传的文件。
method: 
请求方法，比如POST、GET
知识点参考链接：https://blog.csdn.net/yannanxiu/article/details/53116652
werkzeug
"""

def upload_image():
    global secure_filename
    if flask.request.method == "POST":  # 设置request的模式为POST
        img_file = flask.request.files["image_file"]  # 获取需要分类的图片
        secure_filename = werkzeug.secure_filename(img_file.filename)  # 生成一个没有乱码的文件名
        img_path = os.path.join(app.root_path, "predict_img/"+secure_filename)  # 获取图片的保存路径
        img_file.save(img_path)  # 将图片保存在应用的根目录下
        print("图片上传成功.")

        return flask.redirect(flask.url_for(endpoint="predict"))
    return "图片上传失败"

#增加upload路由，使用POST方法，用于文件的上窜
app.add_url_rule(rule="/upload/", endpoint="upload", view_func=upload_image, methods=["POST"])

def predirect_upload():
    return flask.render_template(template_name_or_list="upload_image.html")

"""
"""
app.add_url_rule(rule="/", endpoint="homepage", view_func=predirect_upload)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8008, debug=False)
