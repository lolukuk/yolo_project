import cv2
import os
import argparse
import re


def extract_frames(video_path, output_dir, interval):
    """
    Извлекает кадры из видеофайла и сохраняет их как изображения,
    продолжая нумерацию с последнего существующего файла.

    Args:
        video_path (str): Путь к исходному видеофайлу.
        output_dir (str): Папка для сохранения извлеченных кадров.
        interval (int): Интервал в секундах для извлечения кадров.
    """
    # --- 1. Проверка и подготовка ---
    if not os.path.exists(video_path):
        print(f"Ошибка: Видеофайл не найден по пути: {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Кадры будут сохранены в: {output_dir}")

    frame_pattern = re.compile(r"frame_(\d{5})\.jpg")
    max_frame_num = 0

    existing_files = os.listdir(output_dir)
    if existing_files:
        frame_numbers = []
        for filename in existing_files:
            match = frame_pattern.match(filename)
            if match:
                # Если имя файла соответствует шаблону, извлекаем номер
                frame_numbers.append(int(match.group(1)))
        if frame_numbers:
            max_frame_num = max(frame_numbers)
            print(f"Обнаружены существующие кадры. Максимальный номер: {max_frame_num}.")

    # Новый счетчик будет начинаться со следующего номера
    start_count = max_frame_num + 1
    print(f"Нумерация новых кадров начнется с {start_count:05d}.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видеофайл: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = int(fps * interval)
    if frame_interval <= 0:
        print("Предупреждение: интервал или FPS привели к нулевому шагу. Установка шага в 1 кадр.")
        frame_interval = 1

    print(f"Видео: {os.path.basename(video_path)}")
    print(f"FPS: {fps:.2f}, Всего кадров: {frame_count_total}")
    print(f"Сохраняем 1 кадр каждые {interval} секунд (каждый ~{frame_interval}-й кадр).")
    print("-" * 30)

    current_frame_pos = 0

    saved_frame_idx = start_count

    while True:
        success, frame = cap.read()
        if not success:
            break

        if current_frame_pos % frame_interval == 0:
            filename = f"frame_{saved_frame_idx:05d}.jpg"
            output_path = os.path.join(output_dir, filename)

            cv2.imwrite(output_path, frame)
            print(f"Сохранен кадр {current_frame_pos} как {filename}")

            saved_frame_idx += 1

        current_frame_pos += 1

    cap.release()
    newly_saved_count = saved_frame_idx - start_count

    print("-" * 30)
    print(f"Готово. Всего обработано {current_frame_pos} кадров.")
    print(f"Сохранено {newly_saved_count} новых изображений.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Скрипт для извлечения кадров из видеофайла.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--video_path',
        type=str,
        required=True,
        help='Путь к исходному видеофайлу. \nПример: data/raw_video/my_video.mp4'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Папка, куда будут сохранены кадры. \nПример: data/raw_frames'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=2,
        help='Интервал в секундах между сохраняемыми кадрами. \nПо умолчанию: 2'
    )

    args = parser.parse_args()

    extract_frames(args.video_path, args.output_dir, args.interval)