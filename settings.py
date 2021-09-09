import os
env = os.getenv('EXEC_MODE') or 'dev'

if env == 'dev':
    import dotenv
    dotenv.load_dotenv()

LOG_LEVEL = os.getenv('LOG_LEVEL') or 'DEBUG'
CLOUDKARAFKA_BROKERS = os.getenv('CLOUDKARAFKA_BROKERS')
CLOUDKARAFKA_USERNAME = os.getenv('CLOUDKARAFKA_USERNAME')
CLOUDKARAFKA_PASSWORD = os.getenv('CLOUDKARAFKA_PASSWORD')
CLOUDKARAFKA_TOPIC_PREFIX = os.getenv('CLOUDKARAFKA_TOPIC_PREFIX')
FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY') or 'mySecretKey@'
KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
KAGGLE_KEY = os.getenv('KAGGLE_KEY')
NEXT_URL = os.getenv('NEXT_URL') or 'https://photoode-front.herokuapp.com'
PORT = os.getenv('$PORT') or 5003
