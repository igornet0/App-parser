from .create_app import celery_app
# from .database import SessionLocal
# from .models import TrainingTask
import time
import json

@celery_app.task(bind=True)
def train_model_task(self):
    for i in range(100):
        time.sleep(1)
        self.update_state(state='PROGRESS', meta={'progress': i})

    # db = SessionLocal()
    # try:
    #     # Обновляем статус задачи в БД
    #     task = db.query(TrainingTask).filter(TrainingTask.task_id == task_id).first()
    #     task.status = "processing"
    #     db.commit()
        
    #     # Здесь должна быть ваша реальная логика обучения
    #     print(f"Start training model with dataset: {dataset_path}")
    #     print(f"Parameters: {json.dumps(parameters)}")
        
    #     # Имитация долгого обучения
    #     for i in range(10):
    #         time.sleep(1)
    #         self.update_state(state='PROGRESS', meta={'progress': i*10})
        
    #     # Обновляем статус после успешного выполнения
    #     task.status = "completed"
    #     db.commit()
        
    #     return {"status": "success", "task_id": task_id}
    
    # except Exception as e:
    #     task.status = "failed"
    #     db.commit()
    #     raise self.retry(exc=e, countdown=60, max_retries=3)
    
    # finally:
    #     db.close()