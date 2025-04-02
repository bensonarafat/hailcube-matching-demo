import os 

class Config:
    CELERY_BROKER_URL = os.getenv('RABBITMQ_URL', 'amqp://guest:guest@localhost:5672//')
    CELERY_RESULT_BACKEND = os.getenv('RESULT_BACKEND', 'rpc://')
