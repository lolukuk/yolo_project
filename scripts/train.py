from ultralytics import YOLO
import torch


def main():
    """
    Главная функция для запуска обучения модели YOLO.
    """
    model_name = 'yolo11n.pt'

    data_config_path = '../data/data.yaml'

    #TODO подумать как можно оптимизировать этот процесс

    num_epochs = 100

    image_size = 640

    batch_size = 16

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используемое устройство: {device}")

    model = YOLO(model_name)

    model.to(device)

    print("Начинаем обучение модели...")

    model.train(
        data=data_config_path,
        epochs=num_epochs,
        imgsz=image_size,
        batch=batch_size,
        # name='yolo11n_food_detection_v1'
    )

    print("Обучение завершено.")
    print(f"Результаты сохранены в папке 'runs/train/'.")


if __name__ == '__main__':
    main()