import nltk
# nltk.download('punkt')
import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup


file_positive = 'compliment.csv'
file_negative = 'non_compliment.csv'

wordnet_lemmatizer = WordNetLemmatizer()
stopwords = set(w.rstrip() for w in open('stopwords.txt'))

print("Opening files...")
positive_reviews = pd.read_csv(file_positive)
negative_reviews = pd.read_csv(file_negative)

print("Dropping nan values...")
positive_reviews = positive_reviews['Responses'].dropna()
negative_reviews = negative_reviews['Responses'].dropna()

print("Shuffling positive reviews...")
positive_reviews = positive_reviews.sample(frac=1)
print(positive_reviews.shape)
print(negative_reviews.shape)
negative_reviews = negative_reviews[:len(positive_reviews)]
print(positive_reviews.shape)
print(negative_reviews.shape)
# positive_reviews = positive_reviews[:len(negative_reviews)]




def my_tokenizer(s):
	s = s.lower()
	tokens = nltk.tokenize.word_tokenize(s)
	tokens = [t for t in tokens if len(t)>2]
	tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
	tokens = [t for t in tokens if t not in stopwords]
	return tokens

word_index_map = {}
current_index = 0

positive_tokenized = []
negative_tokenized = []

print("Tokenizing positive reviews...")
for review in positive_reviews:
	tokens = my_tokenizer(review)
	positive_tokenized.append(tokens)
	for token in tokens:
		if token not in word_index_map:
			word_index_map[token] = current_index
			current_index += 1

print("Tokenizing negative reviews...")
for review in negative_reviews:
	tokens = my_tokenizer(review)
	negative_tokenized.append(tokens)
	for token in tokens:
		if token not in word_index_map:
			word_index_map[token] = current_index
			current_index += 1


def tokens_to_vec(tokens,label):
	x = np.zeros(len(word_index_map)+1)
	for t in tokens:
		i = word_index_map[t]
		x[i] += 1
	x = x/x.sum()
	x[-1] = label
	return x

N = len(positive_tokenized) + len(negative_tokenized)

data = np.zeros((N,len(word_index_map)+1))
i=0
for tokens in positive_tokenized:
	xy = tokens_to_vec(tokens,1)
	data[i,:] = xy
	i+=1

for tokens in negative_tokenized:
	xy = tokens_to_vec(tokens,0)
	data[i,:] = xy
	i+=1

np.random.shuffle(data)
data = data[~np.isnan(data).any(axis=1)]
X = data[:,:-1]
Y = data[:,-1]

Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

# print(Xtrain)
# print(Ytrain)
# print('nans: ',sum(np.isnan(Xtrain)))

model = LogisticRegression()
model.fit(Xtrain,Ytrain)
print("Classification rate:",model.score(Xtest,Ytest))
pred = model.predict(Xtest)

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(Xtrain,Ytrain)
print("Classification rate for Adaboost:",model.score(Xtest,Ytest))

# print(Xtest.shape)
# print(Ytest.shape)
# print(pred.shape)
# out = [Xtest,Ytest,pred]
# np.savetxt('test.csv',out,delimiter=',')
# threshold = 0.5
# for word,index in word_index_map.items():
# 	weight = model.coef_[0][index]
# 	if weight >threshold or weight<-threshold:
# 		print(word,weight)














