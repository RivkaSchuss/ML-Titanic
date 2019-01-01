import numpy as np


def train_data(tags, train, test):
    print("hey")


def read_file(file_name):
    i = 0
    attributes = []
    for line in file_name:
        if not i:
            i = 1
            # make attributes
            array = line.split('\t')
            for j in range(len(array)):
                if j == len(array) - 1:
                    attributes.append(array[j].strip("\n"))
                else:
                    attributes.append(array[j])
            for k in range(len(attributes)):
                attributes[k] = []
        else:
            # fill the lists of the attributes
            values = line.split('\t')
            for i in range(len(values)):
                if i == len(values) - 1:
                    attributes[i].append(values[i].strip("\n"))
                else:
                    attributes[i].append(values[i])
        tags = []
        for i in range(len(array)):
            if i == len(array) - 1:
                tags.append(array[i].strip("\n"))
            else:
                tags.append(array[i])
    return tags, attributes


if __name__ == '__main__':
    with open("train.txt") as train_file:
        tags, attributes = read_file(train_file)
    with open("test.txt") as test_file:
        tags_test, tests = read_file(test_file)
    train_data(tags, attributes, tests)
