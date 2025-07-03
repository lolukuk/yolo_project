import cv2
from ultralytics import YOLO

MODEL_PATH = '../runs/detect/train/weights/best.pt'

INPUT_VIDEO_PATH = '../data/raw_video/4_1.MOV'

OUTPUT_VIDEO_PATH = '../results/output_video_4_1.mov'


def main():
    print("Загрузка модели...")
    model = YOLO(MODEL_PATH)
    print("Модель успешно загружена.")

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видеофайл {INPUT_VIDEO_PATH}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    print(f"Начало обработки видео. Результат будет сохранен в {OUTPUT_VIDEO_PATH}")

    # Читаем видео кадр за кадром
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True)

        for r in results:
            annotated_frame = r.plot()

        out.write(annotated_frame)


    cap.release()
    out.release()
    print("Обработка завершена.")

if __name__ == '__main__':
    main()