from celery import Celery
from core import settings

celery_app = Celery(
    "worker",
    broker=settings.run.celery_broker_url,
    backend=settings.run.celery_result_backend,
    include=["backend.celery_app.tasks"]
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)