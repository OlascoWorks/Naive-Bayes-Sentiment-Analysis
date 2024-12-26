import json, re, math

training_data = []
with open('dataset.json') as dataset:
    data = json.load(dataset)
    training_data = [(d['text'], d['sentiment']) for d in data]

def cleanse_and_tokenize(text: str):
    return re.sub(r'[^\w\s]', '', text.removeprefix(' ').removesuffix(' ').lower()).split()

class NaiveBayes():
    def __init__(self, data):
        self.words_per_class = {}
        self.class_counts = {}
        self.vocab = set()
        self.data_length = len(data)

    def train(self):
        for phrase, sentiment in training_data:
            words = cleanse_and_tokenize(phrase)
            self.vocab.update(words)
            self.class_counts[sentiment] = self.class_counts.get(sentiment, 0) + 1

            if sentiment not in self.words_per_class:
                self.words_per_class[sentiment] = {}

            for word in words:
                self.words_per_class[sentiment][word] = self.words_per_class[sentiment].get(word, 0) + 1

    def classify(self, text):
        words = cleanse_and_tokenize(text)
        class_probabilities = {}

        for sentiment in self.words_per_class:
            prior_prob = math.log(self.class_counts[sentiment]/self.data_length)
            likelihood = 0

            for word in words:
                count = self.words_per_class[sentiment].get(word, 0)
                total_words = sum(self.words_per_class[sentiment].values())
                laplace_prob = (count +1) / (total_words + len(self.vocab))
                likelihood += math.log(laplace_prob)

            class_probabilities[sentiment] = prior_prob + likelihood

        return max(class_probabilities, key=class_probabilities.get) if any(class_probabilities.values()) != 0 else 'neutral'
    

if __name__ == '__main__':
    nb = NaiveBayes(training_data)
    nb.train()

    print(nb.classify("I really love this hotel. It's great"))
    print(nb.classify("Who created this terrible movie? Why is it so bad?"))
    print(nb.classify("I think it's okay."))
