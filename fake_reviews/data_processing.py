from __future__ import print_function

import os

import gensim
import matplotlib.pyplot as plt
import numpy as np
from six.moves import cPickle as pickle

import util_constants as util

print("Data statistics:")

truthful_reviews, deceptive_reviews = util.get_review_links()

print('Total Number of truthful reviews: ', len(truthful_reviews))
print('Total Number of deceptive reviews: ', len(deceptive_reviews))


def process_data(filePath):
    with open(filePath, "r") as f:
        lines = f.readlines()
        file_voc = []
        file_numWords = 0
        for line in lines:
            cleanedLine = util.clean_str(line)
            cleanedLine = cleanedLine.strip()
            cleanedLine = cleanedLine.lower()
            words = cleanedLine.split(' ')
            file_numWords = file_numWords + len(words)
            file_voc.extend(words)
    return file_voc, file_numWords


all_reviews = truthful_reviews + deceptive_reviews
vocabulary = []
numWords = []
for fileLink in all_reviews:
    file_voc, file_numWords = process_data(fileLink)
    vocabulary.extend(file_voc)
    numWords.append(file_numWords)

vocabulary = set(vocabulary)
vocabulary = list(vocabulary)

print('Total number of files : ', len(numWords))
print('Total number of words in the files : ', sum(numWords))
print('Vocab size : ', len(vocabulary))
print('Average number of words in the files :', sum(numWords) / len(numWords))

print("Visualizing the data in histogram format:")
plt.hist(numWords, 50)
plt.xlabel('Length of sequence')
plt.ylabel('Frequency of words')
plt.show()

w2v_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
wordsVectors = []
unknown = []
for word in vocabulary:
    try:
        vector = w2v_model[word]
        wordsVectors.append(vector)
    except Exception as e:
        unknown.append(word)
        wordsVectors.append(np.random.uniform(-0.25, 0.25, 300))

del w2v_model
wordsVectors = np.asarray(wordsVectors)

print('The number of unknown words is ', len(unknown))

pickle_file = os.path.join("C:/Users/Ajitkumar/Desktop/DK/malware/project", 'vocab_data.pickle')

try:
    f = open(pickle_file, 'wb')
    save = {
        'wordsVectors': wordsVectors,
        'vocabulary': vocabulary,
        'notFoundwords': unknown
    }

    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)

MAX_SEQ_LENGTH = 160


def convertFileToIndexArray(filePath):
    doc = np.zeros(MAX_SEQ_LENGTH, dtype='int32')
    with open(filePath, "r") as f:
        lines = f.readlines()
        indexCounter = 0
        for line in lines:
            cleanedLine = util.clean_str(line)
            cleanedLine = cleanedLine.strip()
            cleanedLine = cleanedLine.lower()
            words = cleanedLine.split(' ')
            for word in words:
                doc[indexCounter] = vocabulary.index(word)
                indexCounter = indexCounter + 1
                if (indexCounter >= MAX_SEQ_LENGTH):
                    break
            if (indexCounter >= MAX_SEQ_LENGTH):
                break
    return doc


totalFiles = len(truthful_reviews) + len(deceptive_reviews)
idsMatrix = np.ndarray(shape=(totalFiles, MAX_SEQ_LENGTH), dtype='int32')
labels = np.ndarray(shape=(totalFiles, 2), dtype='int32')

counter = 0
for filePath in truthful_reviews:
    idsMatrix[counter] = convertFileToIndexArray(filePath)
    counter = counter + 1

for filePath in deceptive_reviews:
    idsMatrix[counter] = convertFileToIndexArray(filePath)
    counter = counter + 1

labels[0:len(truthful_reviews)] = np.array([1, 0])
labels[len(truthful_reviews):totalFiles] = np.array([0, 1])

print("Dataset is split into training,validation and test set in 8:1:1 ratio")
size = idsMatrix.shape[0]
testSize = int(size * 0.1)
shuffledIndex = np.random.permutation(size)
testIndexes = shuffledIndex[0:testSize]
validationIndexes = shuffledIndex[testSize:2 * testSize]
trainIndexes = shuffledIndex[2 * testSize:size]

test_data = idsMatrix[testIndexes]
test_labels = labels[testIndexes]

validation_data = idsMatrix[validationIndexes]
validation_labels = labels[validationIndexes]

train_data = idsMatrix[trainIndexes]
train_labels = labels[trainIndexes]

pickle_file = os.path.join(
    'C:\Users\Ajitkumar\Desktop\DK\malware\project\cnn-spam-review-detection-tf-master\pickle_files',
    'train_data.pickle')

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_data': train_data,
        'train_labels': train_labels,
        'validation_data': validation_data,
        'validation_labels': validation_labels,
        'test_data': test_data,
        'test_labels': test_labels
    }

    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
