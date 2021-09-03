import json
import logging
import base64
from io import BytesIO

from PIL import Image, ImageOps
import flask
from flask_cors import CORS, cross_origin
import numpy as np

from kafka import kafka
from segmentation.xy_segmentation import xy_segmentation
from settings import FLASK_SECRET_KEY, LOG_LEVEL, NEXT_URL

MAX_HEIGHT = 250
MAX_WIDTH = 500

app = flask.Flask(__name__)
app.config['SECRET_KEY'] = FLASK_SECRET_KEY
CORS(app, origins=NEXT_URL)

logging.basicConfig(level=LOG_LEVEL)
producer = kafka.init_producer()


@app.route("/")
def index():
    return "Running!"


@app.route("/segmentation/<session_id>", methods=('POST',))
@cross_origin()
def segment_image(session_id):
    image_str = flask.request.json['image']
    image_data_str = image_str[image_str.find(',') + 1:]
    image_data = bytes(image_data_str, encoding="ascii")
    image = ImageOps.exif_transpose(Image.open(BytesIO(base64.b64decode(image_data))).convert('L'))
    ratio = min(MAX_HEIGHT/image.size[0], MAX_WIDTH/image.size[1], 1)
    new_size = int(ratio*image.size[0]), int(ratio*image.size[1])
    resized_image = image.resize(new_size, Image.ANTIALIAS)
    image_array = np.array(resized_image.getdata(), dtype='uint8').reshape((*resized_image.size[-1::-1], 1))
    segmentation_results, segmentation_structure = xy_segmentation(image_array)
    message = json.dumps({'segmentation_results': [array.tolist() for array in segmentation_results],
                          'segmentation_structure': segmentation_structure.serialize(),
                          'session_id': session_id})
    kafka.send_message(producer, 'segmentation', message)
    return {"status": "sent for processing"}, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=(LOG_LEVEL == 'DEBUG'), port=8003)
