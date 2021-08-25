import json
import logging

from kafka import kafka
from settings import LOG_LEVEL
from parsing.parser import XYParser
from segmentation.xy_segmentation import dict_to_xy_segmentation_results

logging.basicConfig(level=LOG_LEVEL)

consumer = kafka.init_consumer('classification')

logging.info("Listening for new messages")

try:
    while True:

        input_message = kafka.consumer_cycle(consumer)
        if input_message:
            input_json = json.loads(input_message.value())
            predictions_results = input_json['predictions_results']
            segmentation_dict = input_json['segmentation_structure']
            segmentation_structure = dict_to_xy_segmentation_results(segmentation_dict)
            latex_expression = XYParser(predictions_results, segmentation_structure).last_level.parsed_groups[0]
            logging.info(f'parsed latex: {latex_expression}')


except BaseException as e:
    consumer.close()
    logging.info("Consumer successfully closed")
    raise e
