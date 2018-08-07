# Text-Classification

## Sentiment analysis

The dataset contains 40000 tweets of 13 different emotions. The labels are distributed as follows.

![1](https://github.com/divyakrishna-devisetty/Text-Classification/blob/master/sentiment%20analysis/screenshots/1.JPG)

Tweets data is processed and only the relevant data is retained.

![2](https://github.com/divyakrishna-devisetty/Text-Classification/blob/master/sentiment%20analysis/screenshots/2.JPG)

Tweets are then classified using various classification algorithms such as Multinomial Naive Bayes, Random forest classifier, Linear support vector classification and Logistic Regression, classification accuracy is compared.

![3](https://github.com/divyakrishna-devisetty/Text-Classification/blob/master/sentiment%20analysis/screenshots/3.JPG)

Below is the confusion matrix.

![4](https://github.com/divyakrishna-devisetty/Text-Classification/blob/master/sentiment%20analysis/screenshots/4.JPG)

# Further Improvements

<ul>
  <li> Accuracy can be proved by collecting more data and processing the tweets more accurately</li>
  <li> Using various work vectorization models other than tf-idf, such as word2vec or glove.</li>
</ul>

# Fake Reviews

The dataset consists of reviews from mechanical turk and tripadvisor. This model predicts if a particular review is fake or truthful.

The model is trained using tensorflow.

words distributions:

![hist](https://github.com/divyakrishna-devisetty/Text-Classification/blob/master/fake_reviews/screenshots/hist.PNG)

Model Training:

Accuracy:
  
  ![accuracy graph](https://github.com/divyakrishna-devisetty/Text-Classification/blob/master/fake_reviews/screenshots/accuracy%20graph.PNG)
 
Loss:

  ![loss](https://github.com/divyakrishna-devisetty/Text-Classification/blob/master/fake_reviews/screenshots/loss.PNG)







