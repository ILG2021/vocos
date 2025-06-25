import os
import random

import click


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
    all_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".wav"):
                all_files.append(os.path.join(root, file))

    train_data, val_data = split_data(all_files)

    # 写入文件（可选）
    with open('dataset_raw/val', 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(f"{item}\n")

    with open('dataset_raw/train', 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(f"{item}\n")


if __name__ == '__main__':
    make_dataset()
