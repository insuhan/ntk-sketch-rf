import os
import pickle
import numpy as np
import pandas
from zipfile import ZipFile

UCI_DATASET_URLS = {
    'protein': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv',
    'ct': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00206/slice_localization_data.zip',
    'workloads': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00493/datasets.zip',
    'msd': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip',
}

UCI_DATASET_DATAFILE = {
    'protein': 'CASP.csv',
    'ct': 'slice_localization_data.csv',
    'workloads': 'Range-Queries-Aggregates.csv',
    'msd': 'YearPredictionMSD.txt',
}


def get_basepath():
    return "./"


def get_uci_dataset(dataset):

    if dataset not in UCI_DATASET_URLS.keys():
        raise NotImplementedError

    base_dir = os.path.join(get_basepath(), "data")
    file_dir = os.path.join(base_dir, dataset)
    if not os.path.exists(file_dir):
        print(f"creating directory {file_dir}")
        os.makedirs(file_dir)

    url = UCI_DATASET_URLS[dataset]
    data_file = os.path.join(file_dir, UCI_DATASET_DATAFILE[dataset])
    if not os.path.exists(data_file):
        download_file = os.path.split(url)[-1].replace('%20', ' ')
        if not os.path.exists(os.path.join(file_dir, download_file)):
            print(f"downloading file...")
            os.system(f"wget {url} -P {file_dir}/")

        if '.zip' == download_file[-4:]:
            with ZipFile(os.path.join(file_dir, download_file), 'r') as zip_obj:
                zip_obj.printdir()
                zip_obj.extractall(file_dir)

        if dataset == 'workloads':
            os.system(f"mv {file_dir}/Datasets/* {file_dir}/")
            os.system(f"rm -rf {file_dir}/Datasets")

    if dataset == 'msd':
        x = np.array(pandas.read_csv(data_file, header=None))[:, 1:]
    else:
        x = np.array(pandas.read_csv(data_file))

    if dataset in ['protein', 'msd']:
        y = x[:, 0]
        x = x[:, 1:]
    elif dataset in ['ct']:
        y = x[:, -1]
        x = x[:, 1:-1]
    elif dataset in ['workloads']:
        x = np.unique(x[:, 1:], axis=0)
        while (1):
            if not np.any(np.isnan(x)):
                break
            idx = np.where(np.sum(np.isnan(x), axis=1) == 1)[0][0]
            x = np.delete(x, idx, 0)
        y = x[:, -1]
        x = x[:, :-1]
    else:
        y = x[:, -1]
        x = x[:, :-1]

    return x, y


def get_cifar10():

    base_dir = os.path.join(get_basepath(), "data")
    file_dir = os.path.join(base_dir, 'cifar10')
    if not os.path.exists(file_dir):
        print(f"creating directory {file_dir}")
        os.makedirs(file_dir)

    if not os.path.exists(os.path.join(file_dir, "data_batch_1")):
        print(f"cifar-10 dataset downloading...")
        os.system(f"wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -P {file_dir}")
        os.system(f"tar -xvf {file_dir}/cifar-10-python.tar.gz -C {file_dir}")
        os.system(f"mv {file_dir}/cifar-10-batches-py/* {file_dir}")
        os.system(f"rm -rf {file_dir}/cifar-10-batches-py")

    TRAIN_DATA_FILENAMES = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    TEST_DATA_FILENAME = 'test_batch'

    tr_data = []
    tr_labels = []
    for train_data in TRAIN_DATA_FILENAMES:
        with open(os.path.join(file_dir, train_data), 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        labels = data['labels']
        name = data['filenames']
        data = data['data']
        for i in range(len(data)):
            imgname = name[i]
            data1d = data[i]
            data3d = np.reshape(data1d, (3, 32, 32))
            data3d = np.transpose(data3d, (1, 2, 0))

            tr_data.append(data3d.reshape(-1, *data3d.shape))
        tr_labels.append(np.array(labels))

    tr_data = np.concatenate(tr_data)
    tr_labels = np.concatenate(tr_labels)

    with open(os.path.join(file_dir, TEST_DATA_FILENAME), 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    te_labels = data['labels']
    name = data['filenames']
    data = data['data']
    te_data = []
    for i in range(len(data)):
        imgname = name[i]
        data1d = data[i]
        data3d = np.reshape(data1d, (3, 32, 32))
        data3d = np.transpose(data3d, (1, 2, 0))
        te_data.append(data3d.reshape(-1, *data3d.shape))

    te_data = np.concatenate(te_data)
    te_labels = np.array(te_labels)

    tr_data = tr_data / 255
    te_data = te_data / 255

    mean_ = tr_data.reshape(-1, 3).mean(axis=0).reshape(1, 1, 1, -1)
    std_ = tr_data.reshape(-1, 3).std(axis=0).reshape(1, 1, 1, -1)
    tr_data = (tr_data - mean_) / std_
    te_data = (te_data - mean_) / std_

    return tr_data, tr_labels, te_data, te_labels
