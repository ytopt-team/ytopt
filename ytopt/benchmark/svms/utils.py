import csv
import numpy as np
import logging
from numpy.typing import NDArray
from pandas import DataFrame, concat
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from os import walk, path
from typing import List, TypedDict
from static import CLASSES


class TrainData(TypedDict):
    x_train: NDArray[np.int64]
    x_test: NDArray[np.int64]
    y_train: NDArray[np.int64]
    y_test: NDArray[np.int64]


class Utils:
    window_size = 160
    maximum_counting = 10000

    classes = CLASSES
    n_classes = len(classes)
    count_classes = [0] * n_classes

    data: List[List[int]] = []

    def __init__(self, data_path: str):
        self.data_path = data_path

    def read_data(self) -> List[List[int]]:
        x: List[List[int]] = []
        y: List[int] = []

        records: List[str] = []
        annotations: List[str] = []

        all_file_paths: List[str] = []
        for dirpath, _, filenames in walk(self.data_path):
            for filename in filenames:
                full_path = path.join(dirpath, filename)
                all_file_paths.append(full_path)
        all_file_paths.sort()

        logging.info("Reading data...")
        for file_path in all_file_paths:
            filename, file_extension = path.splitext(file_path)
            if file_extension == ".txt":
                annotations.append(file_path)
            elif file_extension == ".csv":
                records.append(file_path)

        for (record, annotation) in zip(records, annotations):
            signals: List[int] = []
            with open(record, "r") as csvfile:
                reader = csv.reader(
                    csvfile,
                    quotechar="|",
                    delimiter=",",
                )
                next(reader)  # skip header

                for row in reader:
                    signals.append(int(row[1]))

            beat = []
            with open(annotation, "r") as txtfile:
                content = txtfile.readlines()
                for index in range(1, len(content)):
                    splitted = list(
                        filter(lambda v: v != '',
                               content[index].strip().split(' ')))
                    pos = int(splitted[1])
                    arrhythmia_type = splitted[2]

                    if (arrhythmia_type in self.classes):
                        arrhythmia_index = self.classes.index(arrhythmia_type)
                        # avoid overfitting
                        if self.count_classes[
                                arrhythmia_index] > self.maximum_counting:
                            pass
                        else:
                            self.count_classes[arrhythmia_index] += 1
                            if self.window_size < pos < (len(signals) -
                                                         self.window_size):
                                beat = signals[pos - self.window_size + 1:pos +
                                               self.window_size]
                                x.append(beat)
                                y.append(arrhythmia_index)

        self.data = []
        for (i, v) in enumerate(x):
            self.data.append(v)
            self.data[i].append(y[i])

        return self.data

    def get_train_data(self, sample_size: int = 5000) -> TrainData:
        if not self.data:
            self.read_data()

        last_index = len(self.data[0]) - 1

        df = DataFrame(self.data)
        df_1 = df[df[last_index] == 1]
        df_2 = df[df[last_index] == 2]
        df_3 = df[df[last_index] == 3]
        df_4 = df[df[last_index] == 4]
        df_5 = df[df[last_index] == 5]
        df_0 = df[df[last_index] == 0].sample(n=sample_size, random_state=42)

        df_1_upsample = DataFrame(
            resample(df_1,
                     replace=True,
                     n_samples=sample_size,
                     random_state=122))
        df_2_upsample = DataFrame(
            resample(df_2,
                     replace=True,
                     n_samples=sample_size,
                     random_state=123))
        df_3_upsample = DataFrame(
            resample(df_3,
                     replace=True,
                     n_samples=sample_size,
                     random_state=124))
        df_4_upsample = DataFrame(
            resample(df_4,
                     replace=True,
                     n_samples=sample_size,
                     random_state=125))
        df_5_upsample = DataFrame(
            resample(df_5,
                     replace=True,
                     n_samples=sample_size,
                     random_state=126))

        train_df = concat([
            df_0, df_1_upsample, df_2_upsample, df_3_upsample, df_4_upsample,
            df_5_upsample
        ])

        train, test = train_test_split(train_df, test_size=0.2)
        x_train = train.iloc[:, :train.shape[1] - 1].values
        x_test = test.iloc[:, :test.shape[1] - 1].values
        y_train = train[train.shape[1] - 1].values
        y_test = test[test.shape[1] - 1].values

        logging.info(f"Train data size: x {x_train.shape} | y {y_train.shape}")
        logging.info(f"Test data size: x {x_test.shape} | y {y_test.shape}")
        return {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test
        }
