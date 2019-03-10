import socketio
import eventlet
#from eventlet import wsgi
from flask import Flask

from PIL import Image
import numpy as np

import base64
from io import BytesIO

import cv2

from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model = load_model('model.h5')

sio = socketio.Server()

app = Flask(__name__)

gl_speed_limit = 30

# sends control parameters back to the simulator
def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    })

# add preprocessing to images. Same used during learning
def img_preprocess(img):
    # crop scenery and bonnet from images
    # remove everything in 0..60 and in 135..140
    img = img[60:135, :, :]
    # change color space. Use YUV format instead of RGB (Recommended by nVIDIA)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # Gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # resize
    img = cv2.resize(img, (200, 66))
    # normalize
    img = img/255
    return img


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

@sio.on('telemetry')
def telemetry_control(sid, telemetry):
    print(sid)
    cur_speed = float(telemetry['speed'])
    cur_image = Image.open(BytesIO(base64.b64decode(telemetry['image'])))
    cur_image = np.asarray(cur_image)
    # make an array of a single image here for submitting to model
    cur_image = np.array( [img_preprocess(cur_image)] )
    new_steering = float(model.predict(cur_image))

    new_throttle = 1.0 - cur_speed/gl_speed_limit
    print('Speed:{}; Throttle:{}; Steering:{}'.format(cur_speed, new_throttle, new_steering))
    send_control(new_steering, new_throttle)


# Reminder: this shall be placed last in code to avoid problems with callbacks
if __name__ == '__main__':
    # wrap Flask application with socketio's middleware
    app = socketio.Middleware(sio, app)

    try:
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

    except KeyboardInterrupt:
        print('wsgi error')
