import csv
import os
import os.path

import numpy as np
import torch
import torch.utils.data as data

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

urls = {
    'devkit': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar',
    'trainval_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
    'test_images_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
    'test_anno_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar',
}


def read_image_label(file):
    print('[dataset] read ' + file)
    data = dict()
    with open(file, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            name = tmp[0]
            label = int(tmp[-1])
            data[name] = label
            # data.append([name, label])
            # print('%s  %d' % (name, label))
    return data


def read_object_labels(root, dataset, set):
    path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    labeled_data = dict()
    num_classes = len(object_categories)

    for i in range(num_classes):
        file = os.path.join(path_labels, object_categories[i] + '_' + set + '.txt')
        data = read_image_label(file)

        if i == 0:
            for (name, label) in data.items():
                labels = np.zeros(num_classes)
                labels[i] = label
                labeled_data[name] = labels
        else:
            for (name, label) in data.items():
                labeled_data[name][i] = label

    return labeled_data


def write_object_labels_csv(file, labeled_data):
    # write a csv file
    print('[dataset] write file %s' % file)
    with open(file, 'w') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(object_categories)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (name, labels) in labeled_data.items():
            example = {'name': name}
            for i in range(20):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)

    csvfile.close()


def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images


def find_images_classification(root, dataset, set):
    path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    images = []
    file = os.path.join(path_labels, set + '.txt')
    with open(file, 'r') as f:
        for line in f:
            images.append(line)
    return images

class Voc2007Classification(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None):
        if root[-1]=='7':
            dataset='VOC2007'
        else:
            dataset='VOC2012'
        self.dataset = dataset
        self.root = root
        self.path_devkit = os.path.join(root, 'VOCdevkit')
        self.path_images = os.path.join(root, 'VOCdevkit', self.dataset, 'JPEGImages')
        self.set = set
        self.transform = transform
        self.target_transform = target_transform

        # download dataset

        # define path of csv file
        path_csv = os.path.join(self.root, 'files', self.dataset)
        # define filename of csv file
        file_csv = os.path.join(path_csv, 'classification_' + set + '.csv')

        # create the csv file if necessary
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):  # create dir if necessary
                os.makedirs(path_csv)
            # generate csv file
            labeled_data = read_object_labels(self.root, self.dataset, self.set)
            # write csv file
            write_object_labels_csv(file_csv, labeled_data)

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)

