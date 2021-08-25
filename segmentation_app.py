import os
import json
from flask import Flask
from segmentation.xy_segmentation import xy_segmentation
from kafka.kafka import send_message

from settings import FLASK_SECRET_KEY

app = Flask(__name__)
app.config['SECRET_KEY'] = FLASK_SECRET_KEY


@app.route("/")
def index():
    return "Running!"


@app.route("/<session_id>", methods=('POST',))
def segment_image(session_id):
    segmentation_results, segmentation_structure = xy_segmentation()
    message = json.dumps({'segmentation_results': [array.tolist() for array in segmentation_results],
                          'segmentation_structure': segmentation_structure.serialize(),
                          'session_id': session_id})
    send_message('segmentation', message)


if __name__ == '__main__':
    app.run(debug=True)
