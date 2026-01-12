import os
import shutil

base_dir = 'food-101'
images_dir = os.path.join(base_dir, 'images')

dataset_dir = 'dataset'
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

with open(os.path.join(base_dir, 'meta', 'classes.txt'), 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f]

class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

def load_data(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

train_data = load_data(os.path.join(base_dir, 'meta', 'train.txt'))
test_data = load_data(os.path.join(base_dir, 'meta', 'test.txt'))

def prepare_and_copy_data(data, target_split_dir):
    for entry in data:
        class_name, img_name = entry.split('/')
        class_folder = os.path.join(target_split_dir, class_name)
        os.makedirs(class_folder, exist_ok=True)

        src = os.path.join(images_dir, class_name, img_name + '.jpg')
        dst = os.path.join(class_folder, img_name + '.jpg')

        if os.path.isfile(src):
            if not os.path.isfile(dst):
                shutil.copy(src, dst)
            print(f"Kopiuję: {src} -> {dst}")
        else:
            print(f"Brak pliku źródłowego: {src}")

prepare_and_copy_data(train_data, train_dir)
prepare_and_copy_data(test_data, test_dir)

print("Dane zostały skopiowane do struktury Dataset/train i Dataset/test.")
