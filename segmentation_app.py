import os
import json
import logging
import pathlib

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


@app.route("/segmentation/<session_id>", methods=('POST', 'GET'))
def segment_image(session_id):
    base_dir = pathlib.Path(__file__).parents[0]
    img_dir = base_dir / 'dataset' / 'segmentation'
    for img_path in img_dir.iterdir():
        if img_path.is_dir():
            continue
        # path = '/mnt/c/Users/cespa/Desktop/Programming_languages/PhotoODE/segmentation/CodeCogsEqn.png'
        segmentation_results, segmentation_structure = xy_segmentation(img_path)
        message = json.dumps({'segmentation_results': [array.tolist() for array in segmentation_results],
                              'segmentation_structure': segmentation_structure.serialize(),
                              'session_id': session_id})
        kafka.send_message(producer, 'segmentation', message)
    return "{'status': 'sent for processing'}", 200


if __name__ == '__main__':
    app.run(debug=(LOG_LEVEL == 'DEBUG'))
