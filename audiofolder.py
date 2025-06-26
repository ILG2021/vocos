import os
import random
import click
import librosa
import soundfile as sf


def resample_wav(input_path, output_path, target_sr=24000):
    """Resample a WAV file to the target sample rate."""
    # Load audio
    audio, sr = librosa.load(input_path, sr=None)
    # Resample to 24kHz
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save resampled audio
    sf.write(output_path, audio_resampled, target_sr)


def split_data(string_array, val_size=100):
    # 打乱数组
    shuffled_array = string_array.copy()  # 避免修改原数组
    random.shuffle(shuffled_array)

    # 分割数据
    val = shuffled_array[:val_size]
    train = shuffled_array[val_size:]

    return train, val


@click.command()
@click.option("--folder")
def make_dataset(folder):
    # Define output directory for resampled files
    resampled_dir = os.path.join('dataset_raw', 'resampled_24k')

    all_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".wav"):
                input_path = os.path.join(root, file)
                # Create relative path for resampled file
                rel_path = os.path.relpath(input_path, folder)
                output_path = os.path.join(resampled_dir, rel_path)
                # Resample and save
                resample_wav(input_path, output_path)
                all_files.append(output_path)

    train_data, val_data = split_data(all_files)

    # 写入文件
    os.makedirs('dataset_raw', exist_ok=True)
    with open('dataset_raw/val', 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(f"{item}\n")

    with open('dataset_raw/train', 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(f"{item}\n")


if __name__ == '__main__':
    make_dataset()