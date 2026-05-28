# Spam Email Detector

This project is a machine learning spam detector for SMS messages. It classifies each message as either `ham` normal message or `spam` unwanted message.

I built this project to practice a full NLP classification workflow: loading text data, exploring it, cleaning messages, converting text into numerical features, training models, and comparing their results.

## Dataset

The project uses the SMS Spam Collection dataset from the UCI Machine Learning Repository.

The dataset file is included in this repository because it is small and public. It is stored here:

```text
data/spam
```

The original file does not have a `.csv` extension, but it is a tab-separated text file. In the notebook, it is loaded with pandas using `sep="\t"`.

## Tools and Libraries

- Python
- pandas for loading and working with the dataset
- matplotlib for data visualization
- NLTK for stopwords removal
- scikit-learn for TF-IDF vectorization, Multinomial Naive Bayes, Logistic Regression, and evaluation metrics
- Jupyter Notebook

## Project Workflow

First, I loaded the dataset and checked the basic structure of the data. The dataset contains two main columns: the message label and the SMS text.

Then I explored the data by checking the class distribution and message length. The dataset is imbalanced because there are many more normal messages than spam messages. Because of that, I used precision, recall, F1-score, and confusion matrices in addition to accuracy.

For preprocessing, I cleaned the messages by converting text to lowercase, removing punctuation, and removing English stopwords with NLTK.

After that, I converted the cleaned text into numerical features using TF-IDF. I also added a few simple text-based features: message length, number of digits, number of uppercase letters, and number of exclamation marks. These features are useful because spam messages often contain phone numbers, prices, capitalized words, or urgent punctuation.

For TF-IDF, I used `max_features=5000` so the model keeps the most useful words instead of using every rare word in the dataset. I also used `ngram_range=(1, 2)`, which means the model learns both single words and two-word phrases.

This is useful because spam is often easier to recognize from short phrases, not only single words. For example, phrases like `free prize`, `call now`, `urgent reply`, or `claim reward` can give the model stronger signals.

I also added a few extra text features because spam messages often have visible patterns. They may contain phone numbers, prices, many capital letters like `YOU WIN`, exclamation marks, or longer promotional text. These features give the model more information than TF-IDF alone.

## Models

I trained and compared two models:

- Multinomial Naive Bayes
- Logistic Regression

Naive Bayes was used as a strong baseline model for text classification. Logistic Regression was used as a second model because it works well with TF-IDF features and can also use the extra text-based features effectively.

## Results

### Naive Bayes

Naive Bayes achieved:

- accuracy: 97.04%
- spam precision: 1.00
- spam recall: 0.78
- spam F1-score: 0.88

From the confusion matrix:

- 966 normal messages were correctly predicted as normal
- 0 normal messages were incorrectly predicted as spam
- 116 spam messages were correctly predicted as spam
- 33 spam messages were missed and predicted as normal

This means Naive Bayes was very careful when predicting spam. When it predicted spam, it was correct, but it missed more real spam messages.

### Logistic Regression

Logistic Regression achieved:

- accuracy: 97.76%
- spam precision: 0.91
- spam recall: 0.92
- spam F1-score: 0.92

From the confusion matrix:

- 953 normal messages were correctly predicted as normal
- 13 normal messages were incorrectly predicted as spam
- 137 spam messages were correctly predicted as spam
- 12 spam messages were missed and predicted as normal

This means Logistic Regression caught more spam messages than Naive Bayes, while still keeping false spam predictions relatively low.

## Final Model Choice

I selected Logistic Regression as the final model.

For spam detection, recall is very important because the model should catch as many real spam messages as possible. Naive Bayes had perfect spam precision, but it missed 33 spam messages. Logistic Regression missed only 12 spam messages and had a higher spam F1-score.

Because of this, Logistic Regression gave the better balance between detecting spam and avoiding too many false spam predictions.

## Custom SMS Prediction

At the end of the notebook, I added a small function where a user can enter a custom SMS message and get a prediction:

```text
ham
```

or

```text
spam
```

The final demo uses the Logistic Regression model.

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Open the notebook:

```text
notebooks/Spam_Email_Detector.ipynb
```

Run the cells from top to bottom.

## Conclusion

This project shows how a basic NLP spam detector can be built with traditional machine learning. The final Logistic Regression model performed best overall because it detected more real spam messages and achieved the highest spam F1-score.
