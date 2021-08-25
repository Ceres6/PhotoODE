import json
import logging
import pathlib

import numpy as np

from classification.lenet import LeNet
from kafka import kafka
from settings import LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)

consumer = kafka.init_consumer('segmentation')
producer = kafka.init_producer()

base_dir = pathlib.Path(__file__).parents[0]
dataset_dir = base_dir / 'dataset'

# TODO: allow file selection
with open([str(file) for file in dataset_dir.iterdir() if 'label_dict' in str(file)][0], "r") as f:
    labels = list(json.loads(f.read()).values())

logging.debug(f"length of labels {len(labels)}")
lenet = LeNet(labels)
weights_dir = base_dir / 'classification' / 'weights'
weights_file = sorted([path for path in weights_dir.iterdir()])[-1]
lenet.load_weights(weights_file)

logging.info("Listening for new messages")
try:
    while True:

        input_message = kafka.consumer_cycle(consumer)
        if input_message:
            input_json = json.loads(input_message.value())
            segmentation_results = [np.array(image).astype('uint8') for image in input_json['segmentation_results']]
            predictions_results = lenet.predict_array(segmentation_results)
            logging.info(f'prediction results: {predictions_results}')
            output_message = json.dumps({'predictions_results': predictions_results,
                                         'segmentation_structure': input_json['segmentation_structure'],
                                         'session_id': input_json['session_id']})
            kafka.send_message(producer, 'classification', output_message)
            logging.info('classification message sent')

except BaseException as e:
    consumer.close()
    logging.info("Consumer successfully closed")
    raise e
