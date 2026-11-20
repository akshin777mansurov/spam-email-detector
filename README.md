# SMS Spam Detector

This project is a machine learning spam detector for SMS messages. It classifies each message as `ham` (a normal message) or `spam`.

I built it to practice a full NLP classification workflow, and then I went back and validated it properly: I checked the data for leakage, compared the models fairly with cross-validation, chose a decision threshold on purpose, and reported the final result with its uncertainty.

The repository is called `spam-email-detector`, but the data is SMS messages. The same method works for emails.

## Dataset

The project uses the SMS Spam Collection dataset from the UCI Machine Learning Repository.

The dataset file is included in this repository because it is small and public. It is stored here:

```text
data/spam
```

The file has no extension, but it is a tab-separated text file with 5,574 messages.

After removing 403 duplicate messages, 5,171 messages are left: 4,518 ham and 653 spam. Only 12.6% of the messages are spam, so the dataset is imbalanced.

## Tools and Libraries

- Python
- pandas and NumPy for working with the data
- matplotlib for data visualization
- NLTK for stopwords removal
- scikit-learn for TF-IDF, the models, cross-validation and metrics
- Jupyter Notebook

## Project Workflow

1. Load the data and check that every line of the file became one row.
2. Remove duplicate messages before splitting, so no message can be in both the training set and the test set.
3. Explore the class balance and the message lengths.
4. Clean the text: lowercase, remove punctuation and English stopwords, and remove HTML codes left in the dataset.
5. Create features: TF-IDF of the text, plus message length and the number of digits, uppercase letters and exclamation marks.
6. Keep 20% of the data as a test set that is used only once.
7. Compare a baseline, Naive Bayes and Logistic Regression with repeated 5-fold cross-validation on the training set.
8. Tune both models with a grid search.
9. Check for overfitting with training and validation scores and with learning curves.
10. Choose the decision threshold.
11. Evaluate once on the test set, with bootstrap confidence intervals.
12. Read the errors and look at the most important words.

## Validation Choices

**Main metric: PR-AUC.** Because spam is rare, accuracy and ROC-AUC look good for almost any model. A model that always answers `ham` already has 87% accuracy. PR-AUC, precision and recall describe the spam class directly.

**Same features for both models.** Naive Bayes and Logistic Regression get exactly the same input, and both are tuned, so the comparison is fair.

**No leakage inside cross-validation.** TF-IDF and feature scaling are inside a scikit-learn `Pipeline`, so they are fitted only on the training part of each fold.

**Threshold chosen on purpose.** Blocking a normal message is worse than letting a spam message through. So the rule is: precision must be at least 99%, and with that condition recall should be as high as possible. The threshold is chosen from out-of-fold predictions on the training set, not from the test set.

**Final model chosen before the test set.** The final model is the one that catches more spam at the required precision in cross-validation.

## Results

### Cross-validation on the training set

5 folds repeated 3 times, with the default threshold of 0.5. Values are mean ± standard deviation over 15 folds.

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (always ham) | 0.000 | 0.000 | 0.000 | 0.126 | 0.500 |
| Naive Bayes, default | 1.000 ± 0.000 | 0.778 ± 0.051 | 0.874 ± 0.033 | 0.954 ± 0.014 | 0.981 ± 0.006 |
| Logistic Regression, default | 0.941 ± 0.014 | 0.914 ± 0.032 | 0.927 ± 0.019 | 0.972 ± 0.010 | 0.991 ± 0.004 |
| Naive Bayes, tuned | 0.973 ± 0.018 | 0.887 ± 0.040 | 0.927 ± 0.026 | 0.964 ± 0.011 | 0.985 ± 0.005 |
| Logistic Regression, tuned | 0.980 ± 0.012 | 0.900 ± 0.036 | 0.938 ± 0.021 | 0.975 ± 0.009 | 0.990 ± 0.005 |

What the tuning showed:

- For Naive Bayes, the smoothing parameter `alpha` matters a lot. The default `alpha=1.0` misses 22% of spam. The best value is `alpha=0.1`, and `alpha=2.0` is the worst value tested.
- Two-word phrases (bigrams) change PR-AUC by about 0.002 or less. That is smaller than the noise between folds, so they do not really help on this dataset.
- `class_weight="balanced"` does not improve PR-AUC for Logistic Regression. It mostly moves the probabilities up, which has the same effect as a lower threshold.

### Overfitting check

Both tuned models score almost perfectly on their training folds (F1 0.985 and 1.000) and lower on validation folds (F1 0.927 and 0.938). So there is a train-validation gap of about 0.06.

This gap is expected with about 5,000 text features and only about 3,300 training messages per fold. It does not damage the results on new data: stronger regularization gave lower validation scores, not higher, and the test results match the cross-validation results.

The learning curves are almost flat after about 1,500 messages, so more data of the same kind would not help much.

### Threshold and final model

At 99% precision in cross-validation, Naive Bayes catches 87.2% of spam (threshold 0.672) and Logistic Regression 86.4% (threshold 0.839). This is a difference of 4 messages out of 522, so the models are practically tied.

I chose Naive Bayes as the final model. It wins by the rule, and when two models are this close I prefer the simpler one.

### Test set

The test set has 1,035 messages (904 ham and 131 spam). It was used once, after all decisions were made.

| Model | Accuracy | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Baseline (always ham) | 0.873 | 0.000 | 0.000 | 0.000 | 0.127 |
| **Naive Bayes, tuned (final)** | 0.980 | 0.982 | 0.855 | 0.914 | 0.969 |
| Logistic Regression, tuned | 0.984 | 1.000 | 0.870 | 0.931 | 0.979 |

Final model, from the confusion matrix:

- 902 normal messages were correctly kept
- 2 normal messages were wrongly blocked
- 112 spam messages were caught
- 19 spam messages were missed

95% bootstrap confidence intervals for the final model:

| Metric | Test score | 95% interval |
|---|---|---|
| Precision | 0.982 | 0.956 to 1.000 |
| Recall | 0.855 | 0.794 to 0.914 |
| F1 | 0.914 | 0.878 to 0.950 |
| PR-AUC | 0.969 | 0.948 to 0.986 |

Notes on these results:

- Precision is 0.982, slightly below the 99% target. With only 114 messages marked as spam, one wrongly blocked message changes precision by almost one point, so a test set of this size cannot confirm the target exactly.
- The model catches roughly 80-90% of spam. The test set is too small to say more than that.
- Logistic Regression scored slightly better on the test set, but the difference is 2 blocked messages and 2 missed spam messages, well inside the confidence intervals. I did not switch models, because choosing a model after looking at the test set would make the test score unreliable.

## What I Fixed After Validating the First Version

The first version of this project reported 97.0% accuracy for Naive Bayes and 97.8% for Logistic Regression. When I checked how those numbers were produced, I found these problems:

| Problem in the first version | What I did |
|---|---|
| pandas merged some lines because of a `"` character, so 5,572 rows were loaded instead of 5,574 | Load with `quoting=csv.QUOTE_NONE` and check the row count |
| 403 duplicate messages. About one in five spam messages in the test set was also in the training set | Remove duplicates before splitting |
| HTML codes such as `&lt;#&gt;` appear only in ham messages, so the model could use them as a shortcut | Convert them back to characters and remove them |
| Logistic Regression had extra features and class weights, Naive Bayes had neither, and nothing was tuned | Same features for both models, both tuned by grid search |
| One train/test split and no baseline | Repeated cross-validation, an "always ham" baseline, and one final test |
| Accuracy as the headline metric on imbalanced data | PR-AUC, precision and recall for the spam class |
| The README said two-word phrases help, but this was never measured | Measured in the grid search: no real effect |
| Default threshold of 0.5 | Threshold chosen with a precision rule |
| The demo message "Congratulations! You won a free prize. Call now to claim." was classified as ham | Now classified as spam |

## Limitations

- **Small dataset.** Only 653 spam messages are left after removing duplicates, and 131 of them are in the test set. This is why the confidence intervals are wide.
- **Old and local data.** The messages were collected before 2012, mostly in the UK and Singapore. Words like `ringtone` and `nokia` are typical for that time. The model would need retraining for today's spam.
- **No dates.** Without timestamps I could not train on older messages and test on newer ones, so I cannot show how fast the model becomes outdated.
- **Hidden numbers.** In normal messages, the dataset creators replaced some numbers with `<#>`. The digit count may look more useful here than it would be in real life.
- **Similar spam messages.** I removed exact duplicates only. Spam campaigns reuse the same text with a different phone number, so some test spam is still very similar to training spam.
- **Bag of words.** The model counts words and ignores their order and meaning. Spam written in everyday language is often missed.

## Custom SMS Prediction

At the end of the notebook there is a small function that classifies a new SMS message with the final model and the chosen threshold. It returns the label and the spam probability:

```python
predict_sms("Congratulations! You won a free prize. Call now to claim.")
# ('spam', 1.0)
```

## How to Run

The project was tested with Python 3.12.

Install the required libraries:

```bash
pip install -r requirements.txt
```

Open the notebook:

```text
Spam_Email_Detector.ipynb
```

Run the cells from top to bottom. The whole notebook takes about one minute.
