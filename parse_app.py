import concurrent.futures
import json
import logging
import time

import flask
from flask_cors import CORS, cross_origin
from kafka import kafka
from parsing.parser import XYParser
from segmentation.xy_segmentation import dict_to_xy_segmentation_results
from settings import LOG_LEVEL, FLASK_SECRET_KEY, NEXT_URL

app = flask.Flask(__name__)
app.config['SECRET_KEY'] = FLASK_SECRET_KEY
CORS(app, supports_credentials=True)

logging.basicConfig(level=LOG_LEVEL)
consumer = kafka.init_consumer('classification')
parsed_equations = dict()


def message_processor():
    logging.info("Listening for new messages")
    while True:
        input_message = kafka.consumer_cycle(consumer)
        if input_message:
            input_json = json.loads(input_message.value())
            predictions_results = input_json['predictions_results']
            segmentation_dict = input_json['segmentation_structure']
            session_id = input_json['session_id']
            segmentation_structure = dict_to_xy_segmentation_results(segmentation_dict)
            latex_expression = XYParser(predictions_results, segmentation_structure).last_level.parsed_groups[0]
            global parsed_equations
            parsed_equations[session_id] = latex_expression
            logging.info(f'parsed latex: {latex_expression}')
            time.sleep(0.1)



@app.route("/parsed/<session_id>", methods=('GET',))
@cross_origin(supports_credentials=True, origins=NEXT_URL)
def parsed(session_id):
    logging.debug('req received')

    def event_stream():
        logging.debug('event stream')
        global parsed_equations
        while True:
            latex_equation = parsed_equations.get(session_id)
            if latex_equation:
                logging.debug('got it')
                parsed_equations.pop(session_id)
                return f'data: {latex_equation}\n\n'
            time.sleep(0.1)

    return flask.Response(event_stream(), mimetype="text/event-stream")


@app.route("/test")
def test():
    return 'ok', 200


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(message_processor)
        app.run(debug=(LOG_LEVEL == 'DEBUG'), port=5003, threaded=True)

