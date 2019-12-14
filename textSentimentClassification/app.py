
# coding=utf-8
from flask import Flask, render_template, request, make_response
from flask import jsonify
import time
import threading

"""
定义心跳检测函数

"""

def heartbeat():
    print (time.strftime('%Y-%m-%d %H:%M:%S - heartbeat', time.localtime(time.time())))
    timer = threading.Timer(60, heartbeat)
    timer.start()
timer = threading.Timer(60, heartbeat)
timer.start()


app = Flask(__name__,static_url_path="/static") 

@app.route('/message', methods=['POST'])

#"""定义应答函数，用于获取输入信息并返回相应的答案"""
def reply():
#从请求中获取参数信息
    req_msg = request.form['msg']
    res_msg = execute.predict(req_msg)
    return jsonify( { 'text': res_msg } )

@app.route("/")
def index(): 
    return render_template("index.html")
import execute

# 启动APP
if (__name__ == "__main__"): 
    app.run(host = '0.0.0.0', port = 8808) 
