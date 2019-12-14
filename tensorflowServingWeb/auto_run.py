# coding=utf-8
#导入所需要的依赖包
import os
import signal
import subprocess
import getConfig

gConfig={}
gConfig=getConfig.get_config()
tf_model_server=''
flask_server=''

port=gConfig['server_port']
exeport_dir=gConfig['exeport_dir']
try:
    #先启动TensorFlow Serving服务，将模型完成部署
    tf_model_server = subprocess.Popen(["tensorflow_model_server "
                                     "--model_base_path=/home/tf2.0-coding/tensorflowServingWeb/serving_model/ "
                                    "--rest_api_port=9000  --model_name=ImageClassifier"],
                                    stdout=subprocess.DEVNULL,
                                    shell=True,
                                    preexec_fn=os.setsid)
    print("TensorFlow Serving 服务启动成功")
    #启动一个flask 用于api 代理服务器
    flask_server = subprocess.Popen(["export FLASK_ENV=development && flask run --host=0.0.0.0"],
                                    stdout=subprocess.DEVNULL,
                                    shell=True,
                                    preexec_fn=os.setsid)
    print("Flask启动成功")
    #以下是实现退出机制,保证同时退出tensorflow serving服务和flask服务
    while True:
        print("输入q或者exit，回车退出运行程序: ")
        in_str = input().strip().lower()
        if in_str == 'q' or in_str == 'exit':
            print('停止所有服务...')
            os.killpg(os.getpgid(tf_model_server.pid), signal.SIGTERM)
            os.killpg(os.getpgid(flask_server.pid), signal.SIGTERM)
            print('服务停止成功！')
            break
        else:
            continue
except KeyboardInterrupt:
    print('停止所有服务中。。。')
    os.killpg(os.getpgid(tf_model_server.pid), signal.SIGTERM)
    os.killpg(os.getpgid(flask_server.pid), signal.SIGTERM)
    print('所有服务停止成功')
