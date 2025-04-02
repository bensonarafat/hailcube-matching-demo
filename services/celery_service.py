from celery import Celery

# Initialize Celery - don't reference __name__ here
celery = Celery('talent_matching')

def init_celery(app=None):
    # Configure Celery using Flask app settings if provided
    if app:
        celery.conf.update(
            broker_url=app.config['CELERY_BROKER_URL'],
            result_backend=app.config['CELERY_RESULT_BACKEND']
        )
    
    # Add Flask app context to tasks
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            if app:
                with app.app_context():
                    return self.run(*args, **kwargs)
            return self.run(*args, **kwargs)
    
    celery.Task = ContextTask
    return celery
