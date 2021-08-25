from kafka import kafka

consumer = kafka.init_consumer('segmentation')

while True:
    message = kafka.consumer_cycle(consumer)
    if message:
        print('message received')
        break
