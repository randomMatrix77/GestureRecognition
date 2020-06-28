import numpy as np
import cv2
import sklearn
import os

def import_data():
    path = './leap_motion/train_data/'

    folders = os.listdir(path)

    palm = []
    palm_label = []
    fist = []
    fist_labels = []
    other = []
    other_labels = []

    for folder in folders:
        files = os.listdir(os.path.join(path, folder))

        print('Reading files from {} .'.format(folder))

        for file in files:

            img = cv2.imread(os.path.join(path, folder, file), 0)
            img = cv2.resize(img, (224, 224))

            if folder == '0_palm':
                palm.append(img/255)
                palm_label.append([0])
            if folder == '1_fist':
                fist.append(img/255)
                fist_labels.append([1])
            if folder == '2_other':
                other.append(img/255)
                other_labels.append([2])

    palm = sklearn.utils.shuffle(palm)
    fist = sklearn.utils.shuffle(fist)
    other = sklearn.utils.shuffle(other)

    palm_train, palm_val = np.array(palm[:1500]), np.array(palm[1500:])
    palm_train_label, palm_val_label = np.array(palm_label[:1500]), np.array(palm_label[1500:])

    fist_train, fist_val = np.array(fist[:1500]), np.array(fist[1500:])
    fist_train_labels, fist_val_labels = np.array(fist_labels[:1500]), np.array(fist_labels[1500:])

    other_train, other_val = np.array(other[:1500]), np.array(other[1500:])
    other_train_labels, other_val_labels = np.array(other_labels[:1500]), np.array(other_labels[1500:])

    train = np.vstack([palm_train, fist_train, other_train])
    train = np.expand_dims(train, 1)
    train_labels = np.vstack([palm_train_label, fist_train_labels, other_train_labels])

    val = np.vstack([palm_val, fist_val, other_val])
    val = np.expand_dims(val, 1)
    val_labels = np.vstack([palm_val_label, fist_val_labels, other_val_labels])

    train, train_labels = sklearn.utils.shuffle(train, train_labels)
    val, val_labels = sklearn.utils.shuffle(val, val_labels)

    del palm, fist, other
    del palm_train, palm_val, fist_train, fist_val, other_train, other_val, palm_label, fist_labels, other_labels
    del palm_train_label, palm_val_label, fist_train_labels, fist_val_labels, other_train_labels, other_val_labels

    return train, train_labels, val, val_labels
