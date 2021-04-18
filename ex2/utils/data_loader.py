import os
from tqdm import tqdm
import csv
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import zipfile
from pathlib import Path


def load_data(file_path):
    with open(file_path, 'r') as csvfile:
        rows = list(csv.reader(csvfile, delimiter='\t'))[1:]
    return rows


def load_images(dataset_zip_path, rows):
    '''

    :param images_folder:
    :param rows:
    :return: mp arrays of images,y_true, titles
    '''
    first_image_list = []
    second_image_list = []

    first_image_title_list = []
    second_image_title_list = []

    y_true_array = []
    for i, row in tqdm(enumerate(rows)):
        first_image, second_image, y_true, first_image_title, second_image_title = loadImagePairFromRow(
            dataset_zip_path,
            row)
        first_image_list.append(first_image)
        second_image_list.append(second_image)

        y_true_array.append(y_true)

        first_image_title_list.append(first_image_title)
        second_image_title_list.append(second_image_title)

    first_image_list = np.array(first_image_list)
    second_image_list = np.array(second_image_list)
    y_true_array = np.array(y_true_array)
    first_image_title_list = np.array(first_image_title_list)
    second_image_title_list = np.array(second_image_title_list)
    return first_image_list, second_image_list, y_true_array, first_image_title_list, second_image_title_list


def loadImagePairFromRow(dataset_zip_path, row):
    image_title = []
    if len(row) == 3:
        # same
        person_name = row[0]

        first_image_number = row[1]
        second_image_number = row[2]

        first_image = loadImage(dataset_zip_path, person_name, first_image_number)
        first_image_title = person_name + "_" + first_image_number

        second_image = loadImage(dataset_zip_path, person_name, second_image_number)
        second_image_title = person_name + "_" + second_image_number

        return first_image, second_image, 1.0, first_image_title, second_image_title
    else:
        # different
        person_name = row[0]
        first_image_number = row[1]

        second_person_name = row[2]
        second_image_number = row[3]

        first_image_title = person_name + "_" + first_image_number
        second_image_title = second_person_name + "_" + second_image_number

        first_image = loadImage(dataset_zip_path, person_name, first_image_number)
        second_image = loadImage(dataset_zip_path, second_person_name, second_image_number)

        return first_image, second_image, 0.0, first_image_title, second_image_title


def loadImage(images_folder, person_name, image_number):
    filename = r"{0}/{1}/{1}_{2:04d}.jpg".format(images_folder, person_name, int(image_number))
    im = Image.open(filename).convert('L')
    im = np.expand_dims(np.array(im), -1)
    return im  # (250,250,1)


def split_train_datasets(train_dataset, ratio=0.1):
    num_train_samples = int(len(train_dataset) * (1.0 - ratio))

    train = train_dataset[:num_train_samples]
    val = train_dataset[num_train_samples:]
    return train, val


def print_images(row, row_title):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axis = axes.ravel()
    axis[0].imshow(row[0])
    axis[0].set_title(row_title[0])
    axis[1].imshow(row[1])
    axis[1].set_title(row_title[1])
    plt.show()


def load_dataset(dataset_zip_path, train_file, test_file):
    '''

    :param images_folder:
    :param train_file:
    :param test_file:
    :return:
    '''

    path = Path(dataset_zip_path)
    parent_dir = path.parent.absolute()
    images_folder = os.path.join(parent_dir, 'lfw2/lfw2')

    if not os.path.isdir(images_folder):
        with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
            zip_ref.extractall(parent_dir)

    train_rows = load_data(train_file)
    first_image_train_list, second_image_train_list, y_array_train, first_image_title_train_list, second_image_title_train_list = load_images(
        images_folder, train_rows)

    # normalize data
    first_image_train_list = pre_process(first_image_train_list)
    second_image_train_list = pre_process(second_image_train_list)

    train_dataset = [first_image_train_list, second_image_train_list]

    test_rows = load_data(test_file)
    first_image_test_list, second_image_test_list, y_array_test, first_image_title_test_list, second_image_title_test_list = load_images(
        images_folder,
        test_rows)

    # normalize data
    first_image_test_list = pre_process(first_image_test_list)
    second_image_test_list = pre_process(second_image_test_list)
    test_dataset = [first_image_test_list, second_image_test_list]

    return train_dataset, y_array_train, first_image_title_train_list, second_image_title_train_list, \
           test_dataset, y_array_test, first_image_title_test_list, second_image_title_test_list


def pre_process(image_list):
    return image_list / 255.


def split_train_val(train_dataset, y_array_train, ratio=0.1):
    train_ratio = 1.0 - ratio

    total_samples = len(train_dataset[0])
    train_samples = int(total_samples * train_ratio)

    val_dataset = [train_dataset[0][train_samples:], train_dataset[1][train_samples:]]
    y_array_val = y_array_train[train_samples:]

    train_dataset = [train_dataset[0][:train_samples], train_dataset[1][:train_samples]]
    y_array_train = y_array_train[:train_samples]

    return train_dataset, y_array_train, val_dataset, y_array_val

# if __name__ == '__main__':
#     images_folder = r'C:\Users\USER\Desktop\lfwa\lfw2\lfw2'
#     train_file = r'C:\Users\USER\Desktop\lfwa\lfw2\lfw2\pairsDevTrain.txt'
#     test_file = r'C:\Users\USER\Desktop\lfwa\lfw2\lfw2\pairsDevTest.txt'
#     train_dataset, y_array_train, train_titles, test_dataset, y_array_test, test_titles = load_dataset(images_folder,
#                                                                                                        train_file,
#                                                                                                        test_file)
#     print_images(train_dataset[0], train_titles[0])
