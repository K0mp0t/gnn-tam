import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from datetime import datetime


class CustomDataset(Dataset):
    def __init__(self, dataset, window_size=100, step_size=5, mode='train'):
        self.dataset = dataset
        self.window_size = window_size
        self.step_size = step_size
        self.mode = mode

        if dataset in ['IoT_Modbus', 'IoT_Weather']:
            self.dataset_path = os.path.join('./data', dataset + '.csv')
            self.df, self.label = self._prepare_ton_iot()
            self.df, self.label = self._mask_ton_iot()
        elif dataset == 'mhealth':
            self.dataset_path = os.path.join('./data', 'mhealth_dataset')
            self.df, self.label = self._prepare_mhealth()
            self.df, self.label = self._mask_ton_iot()
        elif dataset == 'pamap2':
            self.dataset_path = os.path.join('./data', 'pamap2_dataset', 'Protocol')
            self.df, self.label = self._prepare_pamap2()
            self.df, self.label = self._mask_ton_iot()
        else:
            raise NotImplementedError(f'{dataset} is not supported')

        scaler = StandardScaler()
        self.df.iloc[:] = scaler.fit_transform(self.df)

        self._prepare_window_indices()

    def _prepare_ton_iot(self):
        df = pd.read_csv(self.dataset_path)
        df['datetimes'] = df['date'].apply(lambda x: x.strip()) + ' ' + df['time'].apply(lambda x: x.strip())
        df['datetimes'] = pd.to_datetime(df['datetimes'].apply(lambda x: datetime.strptime(x, '%d-%b-%y %H:%M:%S')))
        df = df.drop(columns=['date', 'time', 'label'])
        df = df.set_index('datetimes')
        df = df.sort_index()

        self.class_mapping = {cls: i for i, cls in enumerate(df['type'].unique())}

        return df.drop(columns=['type']).astype('float32'), df['type'].map(self.class_mapping)

    def _mask_ton_iot(self):
        self.train_mask = pd.Series(False, index=self.df.index, dtype=bool)
        self.test_mask = pd.Series(False, index=self.df.index, dtype=bool)

        cls_windows = dict()

        left, right = 0, 0
        prev = self.label.iloc[0]
        for right in range(1, self.label.shape[0]):
            if self.label.iloc[right] != prev:
                if prev not in cls_windows:
                    cls_windows[prev] = [(left, right)]
                else:
                    cls_windows[prev].append((left, right))
                left = right
            prev = self.label.iloc[right]

        if prev not in cls_windows:
            cls_windows[prev] = [(left, right)]
        else:
            cls_windows[prev].append((left, right))

        cls_windows = {cls: np.concatenate([np.arange(*cls_windows[cls][i]) for i in range(len(cls_windows[cls]))])
                       for cls in cls_windows}

        for cls, cls_indices in cls_windows.items():
            if self.mode == 'train':
                self.train_mask.iloc[cls_indices[:int(len(cls_indices) * 0.8)]] = True
            elif self.mode == 'test':
                self.test_mask.iloc[cls_indices[int(len(cls_indices) * 0.8):]] = True

        if self.mode == 'train':
            return self.df[self.train_mask], self.label[self.train_mask]
        elif self.mode == 'test':
            return self.df[self.test_mask], self.label[self.test_mask]
        else:
            raise NotImplemented(f'{self.mode} is not supported')

    @staticmethod
    def __slice_df_by_label_changepoints(df):
        changepoint_indices = np.where(df.label[:-1].to_numpy() - df.label[1:].to_numpy())[0] + 1
        slices = list()

        left = 0

        for idx in changepoint_indices:
            slices.append(df.iloc[left: idx])
            left = idx

        slices.append(df.iloc[left:])

        return slices
    @staticmethod
    def __merge_slices_by_label(slices):
        grouped_slices = dict()
        for s in slices:
            slice_label = s.iloc[0].label
            if slice_label not in grouped_slices:
                grouped_slices[slice_label] = [s]
            else:
                grouped_slices[slice_label].append(s)

        return [pd.concat(group) for group in list(grouped_slices.values())]

    def _prepare_mhealth(self):
        dataset_dfs = list()

        for fn in os.listdir(self.dataset_path):
            if not fn.endswith('.log'):
                continue
            with open(os.path.join(self.dataset_path, fn)) as f:
                data = f.readlines()
            data = [list(map(float, line.split('\t'))) for line in data]
            data_df = pd.DataFrame(data)

            columns_mapping = {c: f'feature_{c}' if c < len(data_df.columns) - 1 else 'label' for c in data_df.columns}
            data_df = data_df.rename(columns=columns_mapping)

            data_df['label'] = data_df.label.astype(int)
            data_df['subject'] = fn.split('_')[1].split('.')[0]
            dataset_dfs.append(data_df)

        dataset_slices = list()

        for df in dataset_dfs:
            slices = CustomDataset.__slice_df_by_label_changepoints(df)
            slices = CustomDataset.__merge_slices_by_label(slices)

            dataset_slices.extend(slices)

        dataset_slices = CustomDataset.__merge_slices_by_label(dataset_slices)
        assert all([s.label.nunique() == 1 for s in dataset_slices])

        # 0th class has to be downsampled, others are ok
        downsampling_number = sorted([s.shape[0] for s in dataset_slices])[-2]
        zero_class_slice_idx = [s.iloc[0].label for s in dataset_slices].index(0)

        dataset_slices[zero_class_slice_idx] = dataset_slices[zero_class_slice_idx].groupby('subject').head(
            int(downsampling_number / dataset_slices[zero_class_slice_idx].subject.nunique()))

        dataset_df = pd.concat(dataset_slices)
        labels = dataset_df.label
        dataset_df = dataset_df.drop(columns=['label', 'subject'])

        assert labels.nunique() == 13
        assert dataset_df.shape[0] == labels.shape[0]

        return dataset_df.astype('float32'), labels

    def _prepare_pamap2(self):
        dataset_dfs = list()

        for fn in os.listdir(self.dataset_path):
            with open(os.path.join(self.dataset_path, fn)) as f:
                data = f.readlines()

            data = list(map(lambda x: x.strip().split(), data))
            data = list(map(lambda l: list(map(float, l)), data))

            data_df = pd.DataFrame(data, columns=['timestamp', 'label'] + [f'feature_{i}' for i in range(len(data[0]) - 2)])
            data_df['subject'] = fn.split('.')[0]

            dataset_dfs.append(data_df)

        dataset_slices = list()

        for df in dataset_dfs:
            slices = CustomDataset.__slice_df_by_label_changepoints(df)
            slices = CustomDataset.__merge_slices_by_label(slices)

            dataset_slices.extend(slices)

        dataset_slices = CustomDataset.__merge_slices_by_label(dataset_slices)
        assert all([s.label.nunique() == 1 for s in dataset_slices])

        # 0th class has to be downsampled, others are ok
        downsampling_number = sorted([s.shape[0] for s in dataset_slices])[-2]
        zero_class_slice_idx = [s.iloc[0].label for s in dataset_slices].index(0)

        dataset_slices[zero_class_slice_idx] = dataset_slices[zero_class_slice_idx].groupby('subject').head(
            int(downsampling_number / dataset_slices[zero_class_slice_idx].subject.nunique()))

        dataset_df = pd.concat(dataset_slices)
        dataset_df = dataset_df.loc[pd.notna(dataset_df).all(axis=1)]
        labels = dataset_df.label
        labels_mapping = {l: i for i, l in enumerate(sorted(labels.unique()))}
        labels = labels.map(labels_mapping)
        dataset_df = dataset_df.drop(columns=['label', 'subject', 'feature_0', 'timestamp'])

        assert labels.nunique() == 13
        assert dataset_df.shape[0] == labels.shape[0]

        return dataset_df.astype('float32'), labels.astype('int32')

    def _prepare_window_indices(self):
        self.indices = []
        for i in range(0, len(self.df) - self.window_size, self.step_size):
            self.indices.append((i, i + self.window_size))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i, j = self.indices[index]
        return self.df[i:j].to_numpy(), j, max(self.label[i:j].mode())