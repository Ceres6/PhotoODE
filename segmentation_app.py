import os
import json
import logging

from flask import Flask
from kafka import kafka

from segmentation.xy_segmentation import xy_segmentation
from settings import FLASK_SECRET_KEY, LOG_LEVEL

app = Flask(__name__)
app.config['SECRET_KEY'] = FLASK_SECRET_KEY

logging.basicConfig(level=LOG_LEVEL)
producer = kafka.init_producer()


@app.route("/")
def index():
    return "Running!"


@app.route("/<session_id>", methods=('POST', 'GET'))
def segment_image(session_id):
    path = 'C:/Users/cespa/Desktop/Programming_languages/PhotoODE/dataset/segmentation/'
    segmentation_results, segmentation_structure = xy_segmentation(path)
    message = json.dumps({'segmentation_results': [array.tolist() for array in segmentation_results],
                          'segmentation_structure': segmentation_structure.serialize(),
                          'session_id': session_id})
    kafka.send_message(producer, 'segmentation', message)
    return "{'status': 'sent for processing'}", 200


if __name__ == '__main__':
    app.run(debug=(LOG_LEVEL == 'DEBUG'))
