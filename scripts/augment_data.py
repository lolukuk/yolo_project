import os
import cv2
import random
import argparse
import albumentations as A
from tqdm import tqdm


def read_yolo_labels(label_path):
    """Считывает bounding boxes из файла в формате YOLO."""
    if not os.path.exists(label_path):
        return []
    with open(label_path, 'r') as f:
        labels = [line.strip().split() for line in f.readlines()]
        bboxes = []
        for lbl in labels:
            class_id = int(lbl[0])
            coords = [float(c) for c in lbl[1:]]
            bboxes.append([class_id] + coords)
        return bboxes


def write_yolo_labels(label_path, bboxes):
    """Записывает bounding boxes в файл в формате YOLO."""
    with open(label_path, 'w') as f:
        for bbox in bboxes:
            class_id = int(bbox[0])
            coords = ' '.join(map(str, bbox[1:]))
            f.write(f"{class_id} {coords}\n")


def augment_data(source_dir, num_augmentations_per_image=5):
    """
    Применяет аугментацию к изображениям и аннотациям в обучающей выборке.

    Args:
        source_dir (str): Путь к папке с датасетом (например, 'data/processed_dataset').
                          Скрипт будет работать с подпапкой 'train'.
        num_augmentations_per_image (int): Сколько аугментированных версий создавать для каждого исходного изображения.
    """
    train_images_path = os.path.join(source_dir, 'images', 'train')
    train_labels_path = os.path.join(source_dir, 'labels', 'train')

    if not os.path.isdir(train_images_path) or not os.path.isdir(train_labels_path):
        print(f"Ошибка: Папки 'train' не найдены в {source_dir}. Убедитесь, что вы уже разделили датасет.")
        return

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.7,
                           border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Blur(blur_limit=(3, 7), p=0.3),
        A.MotionBlur(blur_limit=(3, 7), p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'],
                                min_area=10, min_visibility=0.1))

    image_files = [f for f in os.listdir(train_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Найдено {len(image_files)} изображений для аугментации.")

    for image_name in tqdm(image_files, desc="Аугментация изображений"):
        base_name = os.path.splitext(image_name)[0]
        image_path = os.path.join(train_images_path, image_name)
        label_path = os.path.join(train_labels_path, f"{base_name}.txt")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = read_yolo_labels(label_path)
        if not bboxes:
            continue

        class_labels = [b[0] for b in bboxes]
        coords = [b[1:] for b in bboxes]

        for i in range(num_augmentations_per_image):
            augmented = transform(image=image, bboxes=coords, class_labels=class_labels)

            aug_image = augmented['image']
            aug_bboxes_coords = augmented['bboxes']
            aug_class_labels = augmented['class_labels']

            if not aug_bboxes_coords:
                continue

            aug_bboxes_full = [[label] + list(coords) for label, coords in zip(aug_class_labels, aug_bboxes_coords)]

            new_base_name = f"{base_name}_aug_{i}"
            new_image_path = os.path.join(train_images_path, f"{new_base_name}.jpg")
            new_label_path = os.path.join(train_labels_path, f"{new_base_name}.txt")

            aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(new_image_path, aug_image_bgr)
            write_yolo_labels(new_label_path, aug_bboxes_full)

    print("\nАугментация завершена!")
    final_count = len(os.listdir(train_images_path))
    print(f"Теперь в обучающей выборке {final_count} изображений.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Офлайн-аугментация для датасета YOLO.")

    parser.add_argument('--source_dir', type=str, required=True,
                        help="Путь к папке 'processed_dataset', содержащей папки 'images/train' и 'labels/train'.")
    parser.add_argument('--num_augmentations', type=int, default=5,
                        help='Количество аугментированных версий на одно исходное изображение (по умолчанию: 5).')

    args = parser.parse_args()
    augment_data(args.source_dir, args.num_augmentations)