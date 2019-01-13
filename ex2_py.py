import math
from collections import Counter


class DtlNode:
    def __init__(self, attribute, depth, is_leaf, pred):
        self.attribute = attribute
        self.depth = depth
        self.is_leaf = is_leaf
        self.pred = pred
        self.children = {}


class DtlTree:
    def __init__(self, root, attributes):
        self.root = root
        self.attributes = attributes

    def tree_string(self, node):
        string = ""
        for child in sorted(node.children):
            string += node.depth * "\t"
            if node.depth > 0:
                string += "|"
            string += node.feature + "=" + child
            if node.children[child].is_leaf:
                string += ":" + node.children[child].pred + "\n"
            else:
                string += "\n" + self.tree_string(node.children[child])

        return string


class DtlClassifier:
    def __init__(self, attributes, examples, tags):
        self.attributes = attributes
        self.examples = examples
        self.tags = tags
        tagged_examples = [(example, tag) for example, tag in zip(self.examples, self.tags)]
        self.tree = DtlTree(self.DTL(tagged_examples, attributes, 0, self.get_mode(tags)), attributes)

    def find_positive_tag(self, tags):
        for tag in tags:
            if tag in ["yes", "true"]:
                return tag
        return tag[0]

    def get_mode(self, tags):
        tags_counter = Counter()
        for tag in tags:
            tags_counter[tag] += 1

        if len(tags_counter) == 2 and list(tags_counter.values())[0] == list(tags_counter.values())[1]:
            return self.find_positive_tag(tags_counter.keys())

        return tags_counter.most_common(1)[0][0]

    def calculate_entropy(self, tags):
        tags_counter = Counter()

        if not tags:
            return 0

        for tag in tags:
            tags_counter[tag] += 1
        class_prob = [tags_counter[tag] / float(len(tags)) for tag in tags_counter]
        if 0.0 in class_prob:
            return 0

        entropy = 0
        for prob in class_prob:
            entropy -= prob * math.log(prob, 2)

        return entropy

    def get_attr_index(self, attr):
        return self.attributes.index(attr)

    def get_attr_domain_dict(self):
        attr_domain_dict = {}
        for attr_index in range(len(self.examples[0])):
            domain = set([ex[attr_index] for ex in self.examples])
            attr_domain_dict[self.attributes[attr_index]] = domain

        return attr_domain_dict

    def get_gain(self, examples, tags, attribute):
        initial_entropy = self.calculate_entropy(tags)
        relative_entropy_per_feature = []
        feature_index = self.get_attr_index(attribute)
        for possible_value in self.get_attr_domain_dict[attribute]:
            examples_and_tags_vi = [(example, tag) for example, tag in zip(examples, tags)
                                    if example[feature_index] == possible_value]
            tags_vi = [tag for example, tag in examples_and_tags_vi]
            entropy_vi = self.calculate_entropy(tags_vi)
            if not examples:
                pass
            relative_entropy = (float(len(examples_and_tags_vi)) / len(examples)) * entropy_vi
            relative_entropy_per_feature.append(relative_entropy)

        return initial_entropy - sum(relative_entropy_per_feature)

    def choose_attribute(self, attributes, examples, tags):
        gain_dict = {attribute: self.get_gain(examples, tags, attribute) for attribute in attributes}
        max_gain = 0
        max_attr = attributes[0]
        for attr in attributes:
            if gain_dict[attr] > max_gain:
                max_gain = gain_dict[attr]
                max_attr = attr
        return max_attr

    def DTL(self, data, attributes, depth, default=None):
        if not len(data):
            return DtlNode(None, depth, True, default)

        tags = [tag for example, tag in data]
        examples = [example for example, tag in data]

        if len(set(tags)) == 1:
            return DtlNode(None, depth, True, tags[0])

        elif not len(attributes):
            return DtlNode(None, depth, True, self.get_mode(tags))
        else:
            best = self.choose_attribute(attributes, examples, tags)


def train_data(features, examples, tags):

    dtl_classifier = DtlClassifier(features[:len(features) - 1], examples, tags)
    print(dtl_classifier.tree.root)
    #
    # classifiers = [dtl_classifier]
    # for classifier in classifiers:
    #     preds = []
    #     for example, tag in zip(examples, tags):
    #         pred = classifier.predict(example)
    #         preds.append(pred)
    #     #preds_per_classifier.append(preds)
        #accuracy_per_classifier.append(ut.get_accuracy(test_tags, preds))

    #write_output_files(tags, preds_per_classifier, accuracy_per_classifier, dtl_classifier)


def read_train_file(file_name):
    attributes = []
    examples = []
    tags = []
    with open(file_name, "r") as file:
        content = file.readlines()
        attributes += content[0].strip("\n").strip().split("\t")

        for line in content[1:]:
            line = line.strip("\n").strip().split("\t")
            example, tag = line[:len(line) - 1], line[-1]
            examples.append(example)
            tags.append(tag)

    return attributes, examples, tags


def write_output_files(test_tags, preds_per_classifier, accuracy_per_classifier, dt_cls):

    dt_preds, knn_preds, nb_preds = preds_per_classifier[0], preds_per_classifier[1], preds_per_classifier[2]
    dt_acc, knn_acc, nb_acc = accuracy_per_classifier[0], accuracy_per_classifier[1], accuracy_per_classifier[2]
    with open("output.txt", "w") as output:
        lines = []
        lines.append("Num\tDT\tKNN\tnaiveBayes")
        i = 1
        for true_tag, dt_pred, knn_pred, nb_pred in zip(test_tags, dt_preds, knn_preds, nb_preds):
            lines.append("{}\t{}\t{}\t{}".format(i, dt_pred, knn_pred, nb_pred))
            i += 1
        lines.append("\t{}\t{}\t{}".format(dt_acc, knn_acc, nb_acc))
        output.writelines("\n".join(lines))

    dt_cls.write_tree_to_file("output_tree.txt")


if __name__ == '__main__':
    attributes, examples, tags = read_train_file("train.txt")
    print(attributes, examples, tags)
    #tags_test, tests = read_file(test_file)
    #print(tags_test, tests)
    train_data(attributes[:len(attributes) - 1], examples, tags)



