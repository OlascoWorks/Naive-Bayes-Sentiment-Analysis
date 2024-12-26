# Naive-Bayes-Sentiment-Analysis

## Project Overview
This project is a simple implementation of a Naive Bayes classifier in Python, designed to perform sentiment analysis. It classifies text into one of three categories: Positive, Negative, or Neutral, using a dataset of labeled examples.

### Key Features:
* Written purely in Python without any external machine learning libraries.
* Mimics the functionality of a Naive Bayes model for educational purposes.
* Demonstrates the fundamental principles of probabilistic classification.

This project is not intended for production use but rather to showcase the workings of the Naive Bayes algorithm and basic NLP preprocessing.


<br></br>
## How It Works
1. Dataset:
* The classifier is trained on a JSON dataset containing labeled examples of text and their respective sentiment labels (Positive, Negative, or Neutral).
* Example of a dataset entry:
```json
{
  "text": "The food was amazing and the service was great.",
  "sentiment": "Positive"
}
```
2. Training:
* The model learns word frequencies for each sentiment class and calculates the probabilities of words occurring in those classes.
* Implements Laplace Smoothing to handle unseen words during classification.

3. Classification:
* Given an input text, the classifier calculates probabilities for each sentiment class using the Naive Bayes formula and assigns the sentiment with the highest probability.


<br></br>
## What Is Naive Bayes?
Naive Bayes is a probabilistic algorithm based on Bayes' Theorem, assuming that the features (words in this case) are independent given the class. While this assumption is rarely true in practice, it simplifies calculations and often provides good results.

Bayes' Theorem:
<p align="center">$P(A∣B) = P(B)/P(B∣A)⋅P(A)$</p>
 
In the context of this project:
* **A**: A sentiment class (e.g., Positive, Negative, Neutral)
* **B**: The given words in the text
* **P(A|B)**: Probability of sentiment 𝐴 given the words 𝐵

Laplace smoothing ensures is added to ensure probability of words never become zero.


<br></br>
## How to Use
*Clone the repository and load the dataset:*
```bash
git clone https://github.com/OlascoWorks/Naive-Bayes-Sentiment-Analysis.git
cd Naive-Bayes-Sentiment-Analysis
```
Ensure dataset.json is in the same directory as the code. The dataset should be a JSON array with "text" and "sentiment" fields.

*Run the Python script:*

```bash
python naive_bayes_sentiment.py
```


## Testing
To test the accuracy of the file, run `test.py`. You can add your own testing datasets by overwriting `test.json`


<br></br>
## Example Usage
Input:
```python
nb = NaiveBayes(training_dataset)

print(nb.classify("I really love this hotel. It's great"))
print(nb.classify("Who created this terrible movie? Why is it so bad?"))
print(nb.classify("I think it's okay."))
```

Output:
```mathematica
Positive
Negative
Neutral
```


<br></br>
## Tuning
Modify the dataset.json file or tweak the training parameters (e.g., add new data) to observe changes in classification and improve the classiffiers accuracy
