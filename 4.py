from flask import Flask, send_file
import os

app = Flask(__name__)

@app.route('/image/<path:filename>')
def send_image(filename):
    # 假设图片存储在static文件夹下
    return send_file('/image/'+ filename)

if __name__ == "__main__":
    app.run(debug=True,port=5001,host='0.0.0.0')