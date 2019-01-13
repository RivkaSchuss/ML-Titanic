import math
from collections import Counter


def find_positive_tag(tags):
    for tag in tags:
        if tag in ["yes", "true"]:
            return tag
    return tag[0]


class DtlNode:
    def __init__(self, attr, depth, leaf=False, pred=None):
        self.attr = attr
        self.depth = depth
        self.leaf = leaf
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
            string += node.attr + "=" + child
            if node.children[child].leaf:
                string += ":" + node.children[child].pred + "\n"
            else:
                string += "\n" + self.tree_string(node.children[child])

        return string

    def traversal(self, ex):
        current_node = self.root
        while not current_node.leaf:
            feature_value = ex[self.attributes.index(current_node.attr)]
            current_node = current_node.children[feature_value]

        return current_node.pred


class DtlClassifier:
    def __init__(self, attributes, examples, tags):
        self.attributes = attributes
        self.examples = examples
        self.tags = tags
        self.att_dom_dict = self.get_attr_domain_dict()
        examples_merged_tags = [(example, tag) for example, tag in zip(self.examples, self.tags)]
        self.tree = DtlTree(self.DTL(examples_merged_tags, attributes, 0, self.get_mode(tags)), attributes)

    def write_to_file(self, output_file_name):
        with open(output_file_name, "w") as output:
            tree_string = self.tree.tree_string(self.tree.root)
            output.write(tree_string[:len(tree_string) - 1])
            
    def predict(self, ex):
        return self.tree.traversal(ex)

    def get_mode(self, tags):
        tags_counter = Counter()
        for tag in tags:
            tags_counter[tag] += 1

        if len(tags_counter) == 2 and list(tags_counter.values())[0] == list(tags_counter.values())[1]:
            return find_positive_tag(tags_counter.keys())

        return tags_counter.most_common(1)[0][0]

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

    def get_attr_domain_dict(self):
        attr_domain_dict = {}
        for attr_index in range(len(self.examples[0])):
            domain = set([ex[attr_index] for ex in self.examples])
            attr_domain_dict[self.attributes[attr_index]] = domain

        return attr_domain_dict

    def get_gain(self, examples, tags, attribute):
        initial_entropy = self.calc_entropy(tags)
        relative_entropy_attr = []
        attr_index = self.attributes.index(attribute)
        for possible_value in self.att_dom_dict[attribute]:
            examples_and_tags_vi = [(example, tag) for example, tag in zip(examples, tags)
                                    if example[attr_index] == possible_value]
            tags_vi = [tag for example, tag in examples_and_tags_vi]
            entropy_vi = self.calc_entropy(tags_vi)
            if not examples:
                pass
            relative_entropy = (float(len(examples_and_tags_vi)) / len(examples)) * entropy_vi
            relative_entropy_attr.append(relative_entropy)

        return initial_entropy - sum(relative_entropy_attr)

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


class KnnClassifer:
    def __init__(self, examples, tags, k=5):
        self.examples = examples
        self.tags = tags
        self.k = k

    def calc_distance(self, ex1, ex2):
        distance = 0
        for attr1, attr2 in zip(ex1, ex2):
            if attr1 != attr2:
                distance += 1
        return distance

    def get_common_tag(self, closest_k):
        tags_counter = Counter()
        for tag in closest_k:
            tags_counter[tag] += 1

        return tags_counter.most_common(1)[0][0]

    def predict(self, ex):
        data = [(example, tag) for example, tag in zip(self.examples, self.tags)]
        distances = []
        for pair in data:
            distance = self.calc_distance(pair[0], ex)
            distances.append((pair, distance))

        closest_k = sorted(distances, key=lambda x: x[1])[:self.k]

        closest_k = [element[0][1] for element in closest_k]
        return self.get_common_tag(closest_k)


class NaiveClassifier:
    def __init__(self, examples, tags):
        self.examples = examples
        self.tags = tags
        self.examples_dict = self.get_examples_dict()
        self.attr_dom_dict = self.get_attr_dom_dict()

    def get_examples_dict(self):
        examples_dict = {}
        for ex, tag in zip(self.examples, self.tags):
            if tag in examples_dict:
                examples_dict[tag].append(ex)
            else:
                examples_dict[tag] = [ex]

        return examples_dict

    def get_attr_dom_dict(self):
        attr_domain_size_dict = {}
        for attr_index in range(len(self.examples[0])):
            domain = set([ex[attr_index] for ex in self.examples])
            attr_domain_size_dict[attr_index] = len(domain)
        return attr_domain_size_dict

    def predict(self, ex):
        pass



def read__file(file_name):
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


def write_files(test_tags, preds, accuracies, dtl_classifier):
    dt_preds, knn_preds, naive_preds = preds[0], preds[1], preds[2]
    dt_acc, knn_acc, naive_acc = accuracies[0], accuracies[1], accuracies[2]
    with open("output.txt", "w") as output:
        lines = ["Num\tDT\tKNN\tnaiveBayes"]
        i = 1
        for true_tag, dt_pred, knn_pred, naive_pred in zip(test_tags, dt_preds, knn_preds, naive_preds):
            lines.append("{}\t{}\t{}\t{}".format(i, dt_pred, knn_pred, naive_pred))
            i += 1
        lines.append("\t{}\t{}\t{}".format(dt_acc, knn_acc, naive_acc))
        output.writelines("\n".join(lines))

    dtl_classifier.write_to_file("output_tree.txt")


def get_accuracy(test_tags, preds):
    good = bad = 0.0
    for test_tag, pred in zip(test_tags, preds):
        if test_tag == pred:
            good += 1
        else:
            bad += 1
        accuracy = float(good)/(good + bad)
    return math.ceil(accuracy * 100) / 100


if __name__ == '__main__':
    attributes, train_examples, train_tags = read__file("train.txt")
    place_holder, test_examples, test_tags = read__file("test.txt")

    dtl_classifier = DtlClassifier(attributes[:len(attributes) - 1], train_examples, train_tags)
    knn_classifier = KnnClassifer(train_examples, train_tags)
    naive_classifier = NaiveClassifier(train_examples, train_tags)

    classifiers = [dtl_classifier, knn_classifier, naive_classifier]
    preds_per_classifier = []
    accuracies = []

    for classifier in classifiers:
        preds = []
        for example, tag in zip(test_examples, test_tags):
            pred = classifier.predict(example)
            preds.append(pred)
        preds_per_classifier.append(preds)
        accuracies.append(get_accuracy(test_tags, preds))

    write_files(test_tags, preds_per_classifier, accuracies, dtl_classifier)


