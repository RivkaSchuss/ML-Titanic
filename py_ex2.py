import math
from collections import Counter
from functools import reduce


# determines if the tag is positive or not
def find_positive_tag(tags):
    for tag in tags:
        if tag in ["yes", "true"]:
            return tag
    return tag[0]


# defines a node in the decision tree
class DtlNode:
    def __init__(self, attr, depth, leaf=False, previous=None):
        self.attr = attr
        self.depth = depth
        self.leaf = leaf
        self.previous = previous
        self.children = {}


# defines a class for the decision tree
class DtlTree:
    def __init__(self, root, attributes):
        self.root = root
        self.attributes = attributes

    # exports the tree as a string, in order to be written to a file
    def tree_string(self, node):
        string = ""
        for child in sorted(node.children):
            string += node.depth * "\t"
            if node.depth > 0:
                string += "|"
            string += node.attr + "=" + child
            if node.children[child].leaf:
                string += ":" + node.children[child].previous + "\n"
            else:
                string += "\n" + self.tree_string(node.children[child])
        return string

    # traverses the tree
    def traversal(self, ex):
        current_node = self.root
        # if the current node is not a leaf
        while not current_node.leaf:
            attr_value = ex[self.attributes.index(current_node.attr)]
            current_node = current_node.children[attr_value]

        return current_node.previous


# the class defining the decision tree classifier
class DtlClassifier:
    def __init__(self, attributes, examples, tags):
        self.attributes = attributes
        self.examples = examples
        self.tags = tags
        self.att_dom_dict = self.get_attr_domain_dict()
        # merging the examples with the tags using zip, creating a list of tuples
        examples_merged_tags = [(example, tag) for example, tag in zip(self.examples, self.tags)]
        self.tree = DtlTree(self.DTL(examples_merged_tags, attributes, 0, self.get_mode(tags)), attributes)

    # writes the decision tree to a file
    def write_to_file(self, output_file_name):
        with open(output_file_name, "w") as output:
            tree_string = self.tree.tree_string(self.tree.root)
            output.write(tree_string[:len(tree_string) - 1])

    # returns the prediction, by traversing the decision tree using an example
    def predict(self, ex):
        return self.tree.traversal(ex)

    # gets the mode of the tags, meaning if yes or no is the majority
    def get_mode(self, tags):
        tags_count = Counter()
        for tag in tags:
            tags_count[tag] += 1

        if len(tags_count) == 2 and list(tags_count.values())[0] == list(tags_count.values())[1]:
            return find_positive_tag(tags_count.keys())

        return tags_count.most_common(1)[0][0]

    # calculates the entropy accuracy using the tags
    def calc_entropy(self, tags):
        tags_count = Counter()
        entropy = 0

        if not len(tags):
            return 0

        for tag in tags:
            tags_count[tag] += 1

        class_prob = [tags_count[tag] / float(len(tags)) for tag in tags_count]
        if 0.0 in class_prob:
            return 0

        for prob in class_prob:
            entropy -= prob * math.log(prob, 2)

        return entropy

    # returns the attribute dictionary with the keys being examples
    def get_attr_domain_dict(self):
        attr_domain_dict = {}
        for attr_index in range(len(self.examples[0])):
            domain = set([ex[attr_index] for ex in self.examples])
            attr_domain_dict[self.attributes[attr_index]] = domain

        return attr_domain_dict

    # returns the gain through the attributes
    def get_gain(self, examples, tags, attribute):
        initial_entropy = self.calc_entropy(tags)
        relative_entropy_attr = []
        attr_index = self.attributes.index(attribute)
        for possible_value in self.att_dom_dict[attribute]:
            data = [(example, tag) for example, tag in zip(examples, tags) if example[attr_index] ==
                                    possible_value]
            tags_vi = [tag for example, tag in data]
            entropy_vi = self.calc_entropy(tags_vi)
            if not examples:
                pass
            relative_entropy = (float(len(data)) / len(examples)) * entropy_vi
            relative_entropy_attr.append(relative_entropy)

        return initial_entropy - sum(relative_entropy_attr)

    # chooses the best attribute in order to progress with the next attribute
    def choose_attribute(self, attributes, examples, tags):
        gain_dict = {attribute: self.get_gain(examples, tags, attribute) for attribute in attributes}
        max_gain = 0
        max_attr = attributes[0]
        for attr in attributes:
            if gain_dict[attr] > max_gain:
                max_gain = gain_dict[attr]
                max_attr = attr
        return max_attr

    # a recursive function to build the decision tree, returns a node
    def DTL(self, data, attributes, depth, default=None):
        if not len(data):
            return DtlNode(None, depth, True, default)

        tags = [tag for ex, tag in data]
        examples = [ex for ex, tag in data]

        if len(set(tags)) == 1:
            return DtlNode(None, depth, True, tags[0])

        elif not len(attributes):
            return DtlNode(None, depth, True, self.get_mode(tags))
        else:
            best = self.choose_attribute(attributes, examples, tags)
            attr_index = self.attributes.index(best)
            current_node = DtlNode(best, depth)
            attributes_child = attributes[:]
            attributes_child.remove(best)
            for possible_value in self.att_dom_dict[best]:
                examples_and_tags_vi = [(example, tag) for example, tag in zip(examples, tags)
                                        if example[attr_index] == possible_value]
                child = self.DTL(examples_and_tags_vi, attributes_child, depth + 1, self.get_mode(tags))
                current_node.children[possible_value] = child
            return current_node


# a class for the knn classifier
class KnnClassifer:
    def __init__(self, examples, tags, k=5):
        self.examples = examples
        self.tags = tags
        self.k = k

    # calculates the hamming distance between 2 examples
    def calc_distance(self, ex1, ex2):
        distance = 0
        for attr1, attr2 in zip(ex1, ex2):
            if attr1 != attr2:
                distance += 1
        return distance

    # gets the common tag within a certain number of closest k's
    def get_common_tag(self, closest_k):
        tags_counter = Counter()
        for tag in closest_k:
            tags_counter[tag] += 1

        return tags_counter.most_common(1)[0][0]

    # predicts the test using the example given
    def predict(self, ex):
        data = [(example, tag) for example, tag in zip(self.examples, self.tags)]
        distances = []
        for pair in data:
            distance = self.calc_distance(pair[0], ex)
            distances.append((pair, distance))

        closest_k = sorted(distances, key=lambda x: x[1])[:self.k]

        closest_k = [element[0][1] for element in closest_k]
        return self.get_common_tag(closest_k)


# a class for the naive bayes classifier
class NaiveClassifier:
    def __init__(self, examples, tags):
        self.examples = examples
        self.tags = tags
        self.examples_dict = self.get_examples_dict()
        self.attr_dom_dict = self.get_attr_dom_dict()

    # returns a dictionary with the examples per tag
    def get_examples_dict(self):
        examples_dict = {}
        for ex, tag in zip(self.examples, self.tags):
            if tag in examples_dict:
                examples_dict[tag].append(ex)
            else:
                examples_dict[tag] = [ex]

        return examples_dict

    # returns the attributes according to the domain size
    def get_attr_dom_dict(self):
        attr_domain_size_dict = {}
        for attr_index in range(len(self.examples[0])):
            domain = set([ex[attr_index] for ex in self.examples])
            attr_domain_size_dict[attr_index] = len(domain)
        return attr_domain_size_dict

    # calculates the probability
    def calc_prob(self, ex, tags):
        conditioned_probs = []
        tags_size = len(tags)
        for attr_index in range(len(ex)):
            count = 1
            domain_size = self.attr_dom_dict[attr_index]
            for train_example in tags:
                if train_example[attr_index] == example[attr_index]:
                    count += 1
            conditioned_probs.append(float(count) / (tags_size + domain_size))
        class_prob = float(len(tags)) / len(self.examples)

        return reduce(lambda x, y: x * y, conditioned_probs) * class_prob

    # predicts through the tests, returns the tag with the highest probability
    def predict(self, ex):
        max_prob = 0
        max_tag = list(self.examples_dict.keys())[0]
        probabilities = []

        # iterates over the dictionary of examples and finds the max tag
        for tag in self.examples_dict:
            probability = self.calc_prob(example, self.examples_dict[tag])
            probabilities.append(probability)
            if probability > max_prob:
                max_prob, max_tag = probability, tag

        if len(probabilities) == 2 and probabilities[0] == probabilities[1]:
            return find_positive_tag(self.examples_dict.keys())

        # returns the greatest tag found
        return max_tag


# reads a file when given a file name
def read__file(file_name):
    attributes = []
    examples = []
    tags = []
    with open(file_name, "r") as file:
        content = file.readlines()
        # strips according to tabs and lines
        attributes += content[0].strip("\n").strip().split("\t")

        for line in content[1:]:
            line = line.strip("\n").strip().split("\t")
            example, tag = line[:len(line) - 1], line[-1]
            examples.append(example)
            tags.append(tag)

    return attributes, examples, tags


# writes all the output to the output file
def write_files(true_tags, predictions, accuracies, dtl_classifier):
    dt_prediction, knn_prediction, naive_prediction = predictions[0], predictions[1], predictions[2]
    dt_accuracy, knn_accuracy, naive_accuracy = accuracies[0], accuracies[1], accuracies[2]
    with open("output.txt", "w") as output:
        # defines the header
        lines = ["Num\tDT\tKNN\tnaiveBase"]
        i = 1
        for true_tag, dt_prediction, knn_prediction, naive_prediction in zip(true_tags, dt_prediction,
                                                                             knn_prediction, naive_prediction):
            lines.append("{}\t{}\t{}\t{}".format(i, dt_prediction, knn_prediction, naive_prediction))
            i += 1
        lines.append("\t{}\t{}\t{}".format(dt_accuracy, knn_accuracy, naive_accuracy))
        output.writelines("\n".join(lines))

    # writes the decision tree to its own file
    dtl_classifier.write_to_file("output_tree.txt")


# calculates the accuracy for each algorithm
def get_accuracy(test_tags, predictions):
    good_accuracy = 0
    bad_accuracy = 0
    for test_tag, prediction in zip(test_tags, predictions):
        if test_tag == prediction:
            good_accuracy += 1
        else:
            bad_accuracy += 1
        accuracy = float(good_accuracy)/(good_accuracy + bad_accuracy)
    return math.ceil(accuracy * 100) / 100


if __name__ == '__main__':
    # parses the 2 files
    attributes, train_examples, train_tags = read__file("train.txt")
    place_holder, test_examples, test_tags = read__file("test.txt")

    # initializes the classifiers for each algorithm
    dtl_classifier = DtlClassifier(attributes[:len(attributes) - 1], train_examples, train_tags)
    knn_classifier = KnnClassifer(train_examples, train_tags)
    naive_classifier = NaiveClassifier(train_examples, train_tags)

    # defines the classifier array
    classifiers = [dtl_classifier, knn_classifier, naive_classifier]
    prediction_per_classifier = []
    accuracies = []

    # defines the predictions
    for classifier in classifiers:
        predictions = []
        for example, tag in zip(test_examples, test_tags):
            # performs the prediction
            prediction = classifier.predict(example)
            predictions.append(prediction)
        prediction_per_classifier.append(predictions)
        # adds the accuracies
        accuracies.append(get_accuracy(test_tags, predictions))

    write_files(test_tags, prediction_per_classifier, accuracies, dtl_classifier)


