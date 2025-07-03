import os
from moviepy import VideoFileClip, concatenate_videoclips


video_paths = [
    '../results/output_video_1.mov',
    '../results/output_video_2_1.mov',
    '../results/output_video_3_1.mov',
    '../results/output_video_3_2.mov',
    '../results/output_video_4.mov',
]

output_filename = 'final_combined_video.mp4'


def combine_videos(file_list, output_path):
    """
    Функция для склейки видеофайлов из списка в один итоговый файл.
    """
    print("Начало процесса склейки видео...")

    for path in file_list:
        if not os.path.exists(path):
            print(f"!!! ОШИБКА: Файл не найден по пути: {path}")
            print("Пожалуйста, проверьте правильность путей в списке 'video_paths' и запустите скрипт снова.")
            return

    try:
        clips = [VideoFileClip(path) for path in file_list]
        print("Клипы успешно загружены.")

        print("Идет процесс склейки...")
        final_clip = concatenate_videoclips(clips, method="compose")

        print(f"Сохранение итогового видео в файл: {output_path}")
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            threads=4  #
        )

        for clip in clips:
            clip.close()

        print("\n" + "=" * 50)
        print(f"Готово! Видео успешно склеено и сохранено как '{output_path}'")
        print("=" * 50)

    except Exception as e:
        print(f"\n!!! Произошла ошибка во время обработки: {e}")
        print("Возможные причины: несовместимость форматов видео, проблемы с кодеками или нехватка памяти.")


if __name__ == '__main__':
    combine_videos(video_paths, output_filename)