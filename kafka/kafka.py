import sys
import os
import logging

from confluent_kafka import Producer, Consumer, KafkaException, KafkaError

from settings import CLOUDKARAFKA_BROKERS, CLOUDKARAFKA_PASSWORD, CLOUDKARAFKA_USERNAME, CLOUDKARAFKA_TOPIC_PREFIX

topic_prefix = CLOUDKARAFKA_TOPIC_PREFIX

# Consumer configuration
# See https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
conf = {
    'bootstrap.servers': CLOUDKARAFKA_BROKERS,
    'session.timeout.ms': 6000,
    'default.topic.config': {'auto.offset.reset': 'smallest'},
    'security.protocol': 'SASL_SSL',
    'sasl.mechanisms': 'SCRAM-SHA-256',
    'sasl.username': CLOUDKARAFKA_USERNAME,
    'sasl.password': CLOUDKARAFKA_PASSWORD
}

producer = Producer(**conf)


def delivery_callback(err, msg):
    if err:
        logging.error(f' Message failed delivery: {err}\n')
    else:
        logging.info(f' Message delivered to {msg.topic()} [{msg.partition()}]\n')


def send_message(topic_suffix: str, message):
    try:
        producer.produce(topic_prefix + topic_suffix, message, callback=delivery_callback)
    except BufferError as e:
        logging.error(f' Local producer queue is full ({len(producer)} messages awaiting delivery): try again\n')
    producer.poll(0)

    logging.error(f' Waiting for {len(producer)} deliveries\n')
    producer.flush()


def init_consumer(topic_suffix: str):
    consumer = Consumer(**conf)
    consumer.subscribe([topic_prefix + topic_suffix])

    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            # Error or event
            if msg.error().code() == KafkaError._PARTITION_EOF:
                # End of partition event
                logging.error(f'{msg.topic()} [{msg.partition()}] reached end at offset {msg.offset()}\n')
            elif msg.error():
                # Error
                raise KafkaException(msg.error())
        else:
            # Proper message
            logging.error(f'{msg.topic()} [{msg.partition()}] at offset {msg.offset()} with key {str(msg.key())}:\n')
            print(msg.value())
