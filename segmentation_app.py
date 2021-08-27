import json
import logging
# import pathlib
import base64
from io import BytesIO

from PIL import Image
import flask
from flask_cors import CORS, cross_origin
import numpy as np

from kafka import kafka
from segmentation.xy_segmentation import xy_segmentation
from settings import FLASK_SECRET_KEY, LOG_LEVEL, NEXT_URL

app = flask.Flask(__name__)
app.config['SECRET_KEY'] = FLASK_SECRET_KEY
CORS(app)

logging.basicConfig(level=LOG_LEVEL)
producer = kafka.init_producer()


@app.route("/")
def index():
    return "Running!"


@app.route("/segmentation/<session_id>", methods=('POST',))
@cross_origin(origins=NEXT_URL)
def segment_image(session_id):
    image_str = flask.request.json['image']
    image_data_str = image_str[image_str.find(',') + 1:]
    image_data = bytes(image_data_str, encoding="ascii")
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image_array = np.array(image.getdata(), dtype='uint8').reshape(image.size[0], image.size[1], 1)
    segmentation_results, segmentation_structure = xy_segmentation(image_array)
    message = json.dumps({'segmentation_results': [array.tolist() for array in segmentation_results],
                          'segmentation_structure': segmentation_structure.serialize(),
                          'session_id': session_id})
    kafka.send_message(producer, 'segmentation', message)
    return {"status": "sent for processing"}, 200


if __name__ == '__main__':
    app.run(debug=(LOG_LEVEL == 'DEBUG'), port=8003)
