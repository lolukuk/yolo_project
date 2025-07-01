import os
import shutil
import random
import argparse


def split_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Разделяет датасет с изображениями и аннотациями в формате YOLO на обучающую,
    валидационную и тестовую выборки. По типу аргументов ниже

    Args:
        source_dir (str): Путь к папке, содержащей все изображения и .txt файлы аннотаций.
        dest_dir (str): Путь к папке, куда будут сохранены разделенные данные.
        train_ratio (float): Доля данных для обучающей выборки.
        val_ratio (float): Доля данных для валидационной выборки.
        test_ratio (float): Доля данных для тестовой выборки.
    """

    if not os.path.isdir(source_dir):
        print(f"Ошибка: Исходная директория не найдена по пути: {source_dir}")
        return

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        print("Ошибка: Сумма долей (train, val, test) должна быть равна 1.")
        return

    print(f"Исходная папка: {source_dir}")
    print(f"Папка назначения: {dest_dir}")
    print(f"Соотношение: Train={train_ratio}, Val={val_ratio}, Test={test_ratio}\n")


    for subset in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_dir, 'images', subset), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'labels', subset), exist_ok=True)

    image_extensions = ('.jpg', '.jpeg', '.png')
    all_files = os.listdir(source_dir)
    image_files = sorted([f for f in all_files if f.lower().endswith(image_extensions)])

    if not image_files:
        print("Ошибка: В исходной директории не найдены изображения.")
        return

    random.shuffle(image_files)

    total_images = len(image_files)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }

    for subset, files in splits.items():
        count = 0
        for image_name in files:
            base_name = os.path.splitext(image_name)[0]
            label_name = f"{base_name}.txt"

            source_image_path = os.path.join(source_dir, image_name)
            source_label_path = os.path.join(source_dir, label_name)

            dest_image_path = os.path.join(dest_dir, 'images', subset, image_name)
            dest_label_path = os.path.join(dest_dir, 'labels', subset, label_name)

            shutil.copy2(source_image_path, dest_image_path)

            if os.path.exists(source_label_path):
                shutil.copy2(source_label_path, dest_label_path)
            else:
                print(f"Предупреждение: Файл аннотации {label_name} не найден для изображения {image_name}.")

            count += 1
        print(f"Скопировано {count} файлов в '{subset}' выборку.")

    print(f"\nРазделение датасета завершено. Всего обработано {total_images} изображений.")
    print(f"Результаты сохранены в: {dest_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Разделение датасета YOLO на train/val/test.")

    parser.add_argument('--source_dir', type=str, required=True,
                        help='Путь к папке с исходными изображениями и аннотациями.')
    parser.add_argument('--dest_dir', type=str, required=True,
                        help='Путь к папке для сохранения структурированного датасета.')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Доля для обучающей выборки (по умолчанию: 0.7).')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Доля для валидационной выборки (по умолчанию: 0.2).')

    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio

    split_dataset(args.source_dir, args.dest_dir, args.train_ratio, args.val_ratio, test_ratio)