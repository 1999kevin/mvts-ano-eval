import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# from sktime.utils import load_data
from ..ucr_anomaly_data_provider.data_loader import load_data
from ..ucr_anomaly_data_provider.loader import Loader
import warnings


warnings.filterwarnings('ignore')

def get_events(y_test, outlier=1, normal=0, breaks=[]):
    events = dict()
    label_prev = normal
    event = 0  # corresponds to no event
    event_start = 0
    for tim, label in enumerate(y_test):
        if label == outlier:
            if label_prev == normal:
                event += 1
                event_start = tim
            elif tim in breaks:
                # A break point was hit, end current event and start new one
                event_end = tim - 1
                events[event] = (event_start, event_end)
                event += 1
                event_start = tim

        else:
            # event_by_time_true[tim] = 0
            if label_prev == outlier:
                event_end = tim - 1
                events[event] = (event_start, event_end)
        label_prev = label

    if label_prev == outlier:
        event_end = tim - 1
        events[event] = (event_start, event_end)
    return events

def format_data(train_df, test_df, OUTLIER_CLASS=1, verbose=False):
    print(train_df)
    print(test_df)
    train_only_cols = set(train_df.columns).difference(set(test_df.columns))
    if verbose:
        print("Columns {} present only in the training set, removing them")
    train_df = train_df.drop(train_only_cols, axis=1)

    test_only_cols = set(test_df.columns).difference(set(train_df.columns))
    if verbose:
        print("Columns {} present only in the test set, removing them")
    test_df = test_df.drop(test_only_cols, axis=1)

    train_anomalies = train_df[train_df["y"] == OUTLIER_CLASS]
    test_anomalies: pd.DataFrame = test_df[test_df["y"] == OUTLIER_CLASS]
    print("Total Number of anomalies in train set = {}".format(len(train_anomalies)))
    print("Total Number of anomalies in test set = {}".format(len(test_anomalies)))
    print("% of anomalies in the test set = {}".format(len(test_anomalies) / len(test_df) * 100))
    print("number of anomalous events = {}".format(len(get_events(y_test=test_df["y"].values))))
    # Remove the labels from the data
    X_train = train_df.drop(["y"], axis=1)
    y_train = train_df["y"]
    X_test = test_df.drop(["y"], axis=1)
    y_test = test_df["y"]
    return X_train, y_train, X_test, y_test

def standardize(X_train, X_test, remove=False, verbose=False, max_clip=5, min_clip=-4):
    mini, maxi = X_train.min(), X_train.max()
    for col in X_train.columns:
        if maxi[col] != mini[col]:
            X_train[col] = (X_train[col] - mini[col]) / (maxi[col] - mini[col])
            X_test[col] = (X_test[col] - mini[col]) / (maxi[col] - mini[col])
            X_test[col] = np.clip(X_test[col], a_min=min_clip, a_max=max_clip)
        else:
            assert X_train[col].nunique() == 1
            if remove:
                if verbose:
                    print("Column {} has the same min and max value in train. Will remove this column".format(col))
                X_train = X_train.drop(col, axis=1)
                X_test = X_test.drop(col, axis=1)
            else:
                if verbose:
                    print("Column {} has the same min and max value in train. Will scale to 1".format(col))
                if mini[col] != 0:
                    X_train[col] = X_train[col] / mini[col]  # Redundant operation, just for consistency
                    X_test[col] = X_test[col] / mini[col]
                if verbose:
                    print("After transformation, train unique vals: {}, test unique vals: {}".format(
                    X_train[col].unique(),
                    X_test[col].unique()))
    return X_train, X_test
    

class PSMSegLoader(Dataset):
    def __init__(self, root_path, win_size=100, step=1, flag="train"):
        root_path = root_path + 'PSM'

        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

        self.name = 'PSM'

        train_df = pd.DataFrame(self.train)
        print(self.test.shape)
        test_df = pd.DataFrame(self.test)
        print(test_df)

        train_df["y"] = np.zeros(train_df.shape[0])
        test_df["y"] = self.test_labels

        X_train, y_train, X_test, y_test = format_data(train_df, test_df, 1, verbose=False)
        X_train, X_test = standardize(X_train, X_test, remove=False, verbose=False)

        self.data = tuple([X_train, y_train, X_test, y_test])
        self.y_test = y_test

    def get_anom_frac_entity(self):
        if self.y_test is None:
            self.load()
        test_anom_frac_entity = (np.sum(self.y_test)) / len(self.y_test)
        return test_anom_frac_entity

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, root_path, win_size=100, step=1, flag="train"):
        root_path = root_path + 'MSL'
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape, ' test labels: ', self.test_labels.shape)
        print("train:", self.train.shape)

        self.name = 'MSL'

        train_df = pd.DataFrame(self.train)
        print(self.test.shape)
        test_df = pd.DataFrame(self.test)
        print(test_df)

        train_df["y"] = np.zeros(train_df.shape[0])
        test_df["y"] = self.test_labels

        X_train, y_train, X_test, y_test = format_data(train_df, test_df, 1, verbose=False)
        X_train, X_test = standardize(X_train, X_test, remove=False, verbose=False)

        self.data = tuple([X_train, y_train, X_test, y_test])
        self.y_test = y_test

    def get_anom_frac_entity(self):
        if self.y_test is None:
            self.load()
        test_anom_frac_entity = (np.sum(self.y_test)) / len(self.y_test)
        return test_anom_frac_entity


    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, root_path, win_size=100, step=1, flag="train"):
        root_path = root_path + 'SMAP'
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

        self.name = 'SMAP'

        train_df = pd.DataFrame(self.train)
        print(self.test.shape)
        test_df = pd.DataFrame(self.test)
        print(test_df)

        train_df["y"] = np.zeros(train_df.shape[0])
        test_df["y"] = self.test_labels

        X_train, y_train, X_test, y_test = format_data(train_df, test_df, 1, verbose=False)
        X_train, X_test = standardize(X_train, X_test, remove=False, verbose=False)

        self.data = tuple([X_train, y_train, X_test, y_test])
        self.y_test = y_test

    def get_anom_frac_entity(self):
        if self.y_test is None:
            self.load()
        test_anom_frac_entity = (np.sum(self.y_test)) / len(self.y_test)
        return test_anom_frac_entity

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, root_path, win_size=100, step=100, flag="train"):
        root_path = root_path + 'SMD'
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

        self.name = 'SMD'

        train_df = pd.DataFrame(self.train)
        print(self.test.shape)
        test_df = pd.DataFrame(self.test)
        print(test_df)

        train_df["y"] = np.zeros(train_df.shape[0])
        test_df["y"] = self.test_labels

        X_train, y_train, X_test, y_test = format_data(train_df, test_df, 1, verbose=False)
        X_train, X_test = standardize(X_train, X_test, remove=False, verbose=False)

        self.data = tuple([X_train, y_train, X_test, y_test])
        self.y_test = y_test

    def get_anom_frac_entity(self):
        if self.y_test is None:
            self.load()
        test_anom_frac_entity = (np.sum(self.y_test)) / len(self.y_test)
        return test_anom_frac_entity

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size=100, step=1, flag="train"):
        root_path = root_path + 'SWaT'
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.name = 'SWAT'

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        self.val = test_data
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

        train_df = pd.DataFrame(self.train)
        print(self.test.shape)
        test_df = pd.DataFrame(self.test)
        print(test_df)

        train_df["y"] = np.zeros(train_df.shape[0])
        test_df["y"] = self.test_labels

        X_train, y_train, X_test, y_test = format_data(train_df, test_df, 1, verbose=False)
        X_train, X_test = standardize(X_train, X_test, remove=False, verbose=False)

        self.data = tuple([X_train, y_train, X_test, y_test])
        self.y_test = y_test

    def get_anom_frac_entity(self):
        if self.y_test is None:
            self.load()
        test_anom_frac_entity = (np.sum(self.y_test)) / len(self.y_test)
        return test_anom_frac_entity

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UCRSegLoader(Dataset):
    def __init__(self, root_path, win_size=100, id=0, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.name = 'UCR'

        DATASET = 'anomaly_archive'  # 'anomaly_archive' OR 'smd'

        root_dir = '/home/liuke/mengxuan/datasets/'
        total_dir = []
        for file in os.listdir(os.path.join(root_dir, 'AnomalyArchive')):
            tmp_dir = '_'.join(file.split('_')[:4])
            total_dir.append(tmp_dir)
        total_dir = sorted(total_dir)

        ENTITY = total_dir[id]  # Name of timeseries in UCR or machine in SMD
        
        train_data_set = load_data(dataset=DATASET, group='train', entities=[ENTITY], downsampling=None, root_dir='root_path', normalize=True, verbose=True)
        self.train = train_data_set.entities[0].Y.transpose((1,0))

        test_data_set = load_data(dataset=DATASET, group='test', entities=[ENTITY], downsampling=None, root_dir='root_path', normalize=True, verbose=True)
        self.test = test_data_set.entities[0].Y.transpose((1,0))
        self.test_labels = test_data_set.entities[0].labels.squeeze(0)
        self.val = self.test

        print("test:", self.test.shape)
        print("train:", self.train.shape)

        train_df = pd.DataFrame(self.train)
        print(self.test.shape)
        test_df = pd.DataFrame(self.test)
        print(test_df)

        train_df["y"] = np.zeros(train_df.shape[0])
        test_df["y"] = self.test_labels

        X_train, y_train, X_test, y_test = format_data(train_df, test_df, 1, verbose=False)
        X_train, X_test = standardize(X_train, X_test, remove=False, verbose=False)

        self.data = tuple([X_train, y_train, X_test, y_test])
        self.y_test = y_test

    def get_anom_frac_entity(self):
        if self.y_test is None:
            self.load()
        test_anom_frac_entity = (np.sum(self.y_test)) / len(self.y_test)
        return test_anom_frac_entity

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
