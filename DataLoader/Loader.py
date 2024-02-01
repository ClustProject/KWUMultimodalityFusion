import numpy as np
import os
import io
import natsort
import torch
import random
from torch_geometric.data import Data, DataLoader

# EEG_band : delta, theta, alpha, beta, gamma, all = 1,2,3,4,5,None
# Feature_name : de_LDS, PSD_LDS, etc.
def load_seedIV_data(data_dir_path: str, feature_name: str, trial: int, islabel=True):
    print("*********** Load features and labels ************")
    print("Feature type : ", feature_name)

    subject_feature_list = np.array([], dtype=float)
    session_dir_list = os.listdir(data_dir_path)

    if islabel:
        subject_label_list = np.array([], dtype=float)
        subject_sample_counts = np.array([], dtype=int)

        label_order = np.array([[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
                                [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
                                [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]])

        for ses_idx, ses_dir in enumerate(session_dir_list):
            session_feature_list, session_label_list, session_sample_counts = np.array([], dtype=float), np.array([],
                                                                                                                  dtype=int), np.array(
                [], dtype=int)

            ses_data_dir_path = data_dir_path + ses_dir + '/'
            file_list = os.listdir(ses_data_dir_path)

            for f_idx, file in enumerate(file_list):
                trial_feature_list, trial_label_list, trial_sample_counts = np.array([], dtype=float), np.array([],
                                                                                                                dtype=int), np.array(
                    [], dtype=int)

                data = io.loadmat(ses_data_dir_path + file)

                for trial_idx in range(1, trial + 1):
                    np_data = data[feature_name + str(trial_idx)][:, :, :]
                    swap_data = np_data.transpose(1, 2, 0)

                    if trial_feature_list.size == 0:
                        trial_feature_list = swap_data.copy()
                    else:
                        trial_feature_list = np.vstack((trial_feature_list, swap_data))

                    trial_label = np.full((swap_data.shape[0]), label_order[ses_idx][trial_idx - 1])
                    trial_label_list = np.hstack((trial_label_list, trial_label))

                    trial_sample_counts = np.hstack((trial_sample_counts, swap_data.shape[0]))

                if session_feature_list.size == 0:
                    session_feature_list = np.expand_dims(trial_feature_list.copy(), axis=0)
                else:
                    session_feature_list = np.vstack((session_feature_list, np.expand_dims(trial_feature_list, axis=0)))

                if session_label_list.size == 0:
                    session_label_list = np.expand_dims(trial_label_list.copy(), axis=0)
                else:
                    session_label_list = np.vstack((session_label_list, np.expand_dims(trial_label_list, axis=0)))

                if session_sample_counts.size == 0:
                    session_sample_counts = np.expand_dims(trial_sample_counts.copy(), axis=0)
                else:
                    session_sample_counts = np.vstack((session_sample_counts, trial_sample_counts))

            if subject_feature_list.size == 0:
                subject_feature_list = session_feature_list
            else:
                subject_feature_list = np.concatenate((subject_feature_list, session_feature_list), axis=1)

            if subject_label_list.size == 0:
                subject_label_list = session_label_list
            else:
                subject_label_list = np.concatenate((subject_label_list, session_label_list), axis=1)

            if subject_sample_counts.size == 0:
                subject_sample_counts = session_sample_counts
            else:
                subject_sample_counts = np.concatenate((subject_sample_counts, session_sample_counts), axis=1)

            print("get DataLoader in session {} ... done".format(ses_idx + 1))
        return subject_feature_list, subject_label_list, subject_sample_counts

    else:
        for ses_idx, ses_dir in enumerate(session_dir_list):
            session_feature_list = np.array([], dtype=float)

            ses_data_dir_path = data_dir_path + ses_dir + '/'
            file_list = os.listdir(ses_data_dir_path)

            for f_idx, file in enumerate(file_list):
                trial_feature_list = np.array([], dtype=float)

                data = io.loadmat(ses_data_dir_path + file)

                for trial_idx in range(1, trial + 1):
                    np_data = data[feature_name + str(trial_idx)][:, :, :]
                    swap_data = np_data.transpose(1, 2, 0)

                    if trial_feature_list.size == 0:
                        trial_feature_list = swap_data.copy()
                    else:
                        trial_feature_list = np.vstack((trial_feature_list, swap_data))

                if session_feature_list.size == 0:
                    session_feature_list = np.expand_dims(trial_feature_list.copy(), axis=0)
                else:
                    session_feature_list = np.vstack((session_feature_list, np.expand_dims(trial_feature_list, axis=0)))

            if subject_feature_list.size == 0:
                subject_feature_list = session_feature_list
            else:
                subject_feature_list = np.concatenate((subject_feature_list, session_feature_list), axis=1)

            print("get DataLoader in session {} ... done".format(ses_idx + 1))
        return subject_feature_list

def load_deap_data(data_dir_path: str, fname1: str, fname2: str, label_dir_path: str, n_columns=2):
    print("*********** Load features and labels ************")

    subject_feature_list1 = np.array([], dtype=float)
    subject_feature_list2 = np.array([], dtype=float)
    subject_label_list = np.array([], dtype=float)
    subject_sample_counts = np.array([], dtype=int)

    feature_dir_list = natsort.natsorted(os.listdir(data_dir_path))

    for feature_idx, feature_dir in enumerate(feature_dir_list):
        feature_data_dir_path = data_dir_path + feature_dir + '/'
        file_list = natsort.natsorted(os.listdir(feature_data_dir_path))
        print(feature_dir)
        feature_list = np.array([], dtype=float)
        if feature_idx == 0:
            feature_name = fname1
        else:
            feature_name = fname2

        for f_idx, file in enumerate(file_list):
            print(file)
            data = io.loadmat(feature_data_dir_path + file)
            np_data = data[feature_name]

            swap_data = np_data.transpose(1, 2, 3, 0)
            shape = swap_data.shape
            swap_data = swap_data.reshape((shape[0] * shape[1], shape[2], shape[3]))

            if feature_list.size == 0:
                feature_list = swap_data.copy()
                feature_list = np.expand_dims(feature_list, axis=0)
            else:
                feature_list = np.vstack((feature_list, np.expand_dims(swap_data, axis=0)))

        if feature_idx == 0:
            print("get data in {} ... done".format(feature_name))
            subject_feature_list1 = feature_list

        else:
            print("get data in {} ... done".format(feature_name))
            subject_feature_list2 = feature_list

    label_list = natsort.natsorted(os.listdir(label_dir_path))

    shape2 = subject_feature_list1.shape
    print("get label ... ", end='')
    subject_label_list = np.zeros((shape2[0], shape2[1], n_columns))
    for l_idx, file in enumerate(label_list):
        print(file)
        label = io.loadmat(label_dir_path + file)
        np_label = label['labels'][:, :n_columns]
        start, end = 0, 0
        for row in np_label:
            end += shape[1]
            label_by_trial = np.full((shape[1], n_columns), row)
            subject_label_list[l_idx][start:end, :] = label_by_trial

            start = end
    print("done")
    return subject_feature_list1, subject_feature_list2, subject_label_list


def deap_label(label_list):
    def binary_label_generator(val):
        if val < 5.:
            return 0
        else:
            return 1

    shape = label_list.shape
    valence_values = label_list[:, 0]
    arousal_values = label_list[:, 1]

    vlc_label = np.zeros(shape[0])
    ars_label = np.zeros(shape[0])

    for i in range(shape[0]):
        vlc_label[i] = binary_label_generator(valence_values[i])
        ars_label[i] = binary_label_generator(arousal_values[i])

    print("************* The number of samples by class ***********")
    print("Threshold : 5.0")
    print("low valence : {},    high valence : {}".format(np.where(vlc_label == 0)[0].shape,
                                                          np.where(vlc_label == 1)[0].shape))
    print("low arousal : {},    high arousal : {}".format(np.where(ars_label == 0)[0].shape,
                                                          np.where(ars_label == 1)[0].shape))

    return vlc_label, ars_label


# edge attributes are composed of edge weights, the distance between all EEG channel pairs
def load_edge_information(pdc_dir_path, sub_name, pdc_var_name, n_channels, n_trials, n_sessions, n_subjects):
    file_list = os.listdir(pdc_dir_path)

    edge_index_list = []
    for i in range(n_channels):
        for j in range(n_channels):
            edge_index_list.append([i, j])

    edge_attr_list = []
    session_edge_attr_list = []
    sub_idx = 0
    for i, file in enumerate(file_list):
        trial_edge_attr_list = []
        pdcs = io.loadmat(pdc_dir_path + file)
        pdc_name = sub_name[sub_idx] + pdc_var_name
        for trial_idx in range(1, n_trials + 1):
            edge_attr = []
            pdc = pdcs[pdc_name + str(trial_idx)][:, :]
            for k in range(n_channels):
                for l in range(n_channels):
                    if k == l:
                        edge_attr.append(0)
                    else:
                        edge_attr.append(pdc[k][l])

            trial_edge_attr_list.append(edge_attr)
        session_edge_attr_list.append(trial_edge_attr_list)
        if (i + 1) % 3 == 0:
            sub_idx += 1
            edge_attr_list.append(session_edge_attr_list)
            session_edge_attr_list = []

    return edge_index_list, edge_attr_list


# Graph Representation
def get_graph_data(subject_data, subject_label, edge_index_list, edge_attr_list, num_train_trials, batch_size):
    edge_index = torch.tensor(edge_index_list, dtype=torch.long)
    train_loader, test_loader = [], []

    num_subjects = len(subject_data)
    num_sessions = len(subject_data[0])
    num_trials = len(subject_data[0][0])

    for subject in range(num_subjects):
        train_dataset, test_dataset = [], []
        for session in range(num_sessions):
            for trial in range(num_trials):
                data_list = []
                trial_data = subject_data[subject][session][
                    trial]  # trial_data = [62][about 240][1or5] = [nodes][trial time][EEG band(s)]
                blocks = len(trial_data[
                                 1])  # about 240(240 sec, 4minutes), 'blocks' is number of blocks which is the same as a trial time
                edge_attr = torch.tensor(edge_attr_list[subject][session][trial], dtype=torch.float)
                for block_idx in range(blocks):
                    # if using features of all frequency bands,
                    # node_feature = [delta, theta, alpha, beta, gamma]
                    data_sample = torch.tensor(trial_data[:, block_idx, :],
                                               dtype=torch.float)  # data_sample = [62][5] = [nodes, node_features(EEG band(s)]
                    data_label = torch.tensor(subject_label[trial], dtype=torch.long)  # data_label [0,2]
                    data_list.append(
                        Data(x=data_sample, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=data_label))
                if trial < num_train_trials:
                    train_dataset.extend(data_list)
                else:
                    test_dataset.extend(data_list)

        random.shuffle(train_dataset)
        random.shuffle(test_dataset)

        batch_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        batch_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        train_loader.append(batch_train_loader)
        test_loader.append(batch_test_loader)
        print('loading' + str(subject))
    print("\nTrain dataset length: {}, \tTest dataset legnth: {}".format(len(train_dataset), len(test_dataset)))
    return train_loader, test_loader