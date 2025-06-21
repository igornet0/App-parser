from backend.celery_app.create_app import celery_app

if __name__ == '__main__':
    celery_app.start()