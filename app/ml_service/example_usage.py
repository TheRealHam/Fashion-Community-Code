from app.ml_service.model_service.base_service import ModelManager
from app.ml_service.model_service.binary_service import evaluate_binary, preprocess_binary, update_binary
from app.ml_service.model_update_schedular.schedular_service import SchedulerService
from app.ml_service.models.binary_classifier_model import BinaryClassifierModel


if __name__ == "__main__":
    # Initialize the model manager
    manager = ModelManager()

    # Register binary classifier
    manager.register_model(
        name="binary",
        model=BinaryClassifierModel(),
        model_path="path/to/binary_model.pth",
        preprocess=preprocess_binary,
        update_fn=update_binary,
        evaluate_fn=evaluate_binary
    )

    # Register other models similarly...

    # Initialize scheduler
    scheduler = SchedulerService()

    # Define daily tasks
    def daily_update_and_evaluate():
        print("Running daily tasks...")
        manager.update_model("binary", new_data_path="path/to/new_data.csv", save_path="path/to/save_model.pth")
        metrics = manager.evaluate_model("binary", validation_data_path="path/to/validation_data.csv")
        print("Evaluation Metrics:", metrics)

    # Schedule the daily task
    scheduler.add_task("daily_update", daily_update_and_evaluate, schedule={'hour': 0, 'minute': 0})

    # Example Prediction
    from PIL import Image
    input_image = Image.open("path/to/user_uploaded_image.jpg")
    prediction = manager.predict("binary", input_image)
    print("Prediction Result:", prediction)

    # Keep the script running
    try:
        while True:
            pass
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
