from naive_bayes import NaiveBayes
import json, math

test_data = []
training_data = []
with open('test.json') as test_dataset, \
      open('dataset.json') as training_dataset:
    data = json.load(test_dataset)
    test_data = [(d['text'], d['sentiment']) for d in data]

    data = json.load(training_dataset)
    training_data = [(d['text'], d['sentiment']) for d in data]

class NaiveBayesTest():
    def __init__(self, test_dataset, training_dataset):
        self.dataset = test_dataset
        self.training_set = training_dataset
        self.class_counts = {}
        self.total_cases = 0
        self.corrects = 0

    def run_test(self):
        nb = NaiveBayes(self.training_set)
        nb.train()

        for phrase, sentiment in self.dataset:
            self.total_cases += 1
            self.class_counts[sentiment] = (self.class_counts.get(sentiment[0], 0) +1, self.class_counts.get(sentiment[1], 0))

            nb_class = nb.classify(phrase)
            if nb_class == sentiment:
                self.corrects += 1
                self.class_counts[sentiment] = (self.class_counts[sentiment][0], self.class_counts[sentiment][1] +1)

        accuracy = {
            "Overall Accuracy:": self.corrects/self.total_cases,
            "Positive Class Accuracy:": self.class_counts['Positive'][1] / self.class_counts['Positive'][0],
            "Negative Class Accuracy:": self.class_counts['Negative'][1] / self.class_counts['Negative'][0],
            "Neutral Class Accuracy:": self.class_counts['Neutral'][1] / self.class_counts['Neutral'][0]
        }

        for label, value in accuracy.items():
            print(f"{label:<28} {value:.2f}")


test = NaiveBayesTest(test_data, training_data)
test.run_test()