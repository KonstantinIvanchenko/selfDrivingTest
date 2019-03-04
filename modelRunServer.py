import socketio
import eventlet
from eventlet import wsgi
from flask import Flask


sio = socketio.Server()

app = Flask(__name__)

if __name__ == '__main__':
    # wrap Flask application with socketio's middleware
    app = socketio.Middleware(socketio_app=app)

    try:
        eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 4679)), app)

    except KeyboardInterrupt:
        print('wsgi error')


@sio.on('connect', namespace='/carsim')
def connect(sid, environ):
    print("connect ", sid)
