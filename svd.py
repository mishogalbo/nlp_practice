import nltk
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

wordnet_lemmatizer = WordNetLemmatizer()

titles = [line.rstrip() for line in open('all_book_titles.txt')]

#load stopwords and add some manual stopwords particular to this analysis
stopwords = set(w.rstrip() for w in open('stopwords.txt'))
stopwords = stopwords.union({
	'introduction','edition','series','application',
	'approach','card','access','package', 'plus', 'etext',
	'third','second','fourth',
	})


def my_tokenizer(s):		###take string and tokenize it
	s = s.lower()
	tokens = nltk.tokenize.word_tokenize(s)		
	tokens = [t for t in tokens if len(t)>2]		#remove words 2 chars or less
	tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]		#get roots/lemmatize
	tokens = [t for t in tokens if t not in stopwords]		#remove stopwords
	tokens = [t for t in tokens if not any(c.isdigit() for c in t)]		#remove numbers
	return tokens

#declare variables
word_index_map = {}		#dictionary, this word is at index n
current_index = 0
all_tokens = []
all_titles = []
index_word_map = []


for title in titles:
	try:
		titles = title.encode('ascii','ignore')		#remove non-ascii chars
		all_titles.append(title)		#add cleaned title to new title array
		tokens = my_tokenizer(title)		#run tokenizer function
		all_tokens.append(tokens)		#add tokens for title to array, array of arrays
		for token in tokens:
			if token not in word_index_map:
				word_index_map[token] = current_index		#if token doesnt exist in map yet, add it to word index map
				current_index += 1
				index_word_map.append(token)		#if token doesnt exist in map yet, add it to index word map
	except:
		pass


def tokens_to_vec(tokens):
	x = np.zeros(len(word_index_map))		#set vector size to N words (tokens)
	for t in tokens:
		x[word_index_map[t]] += 1		#increment value at index of word
	return x

# declare input matrix
X = np.zeros((len(word_index_map),len(all_tokens)))

# create input matrix from tokens
i = 0
for tokens in all_tokens:
	X[:,i] = tokens_to_vec(tokens)
	i += 1


#run SVD
svd = TruncatedSVD(n_components=5)
Z=svd.fit_transform(X)


#create 2d plot
plt.scatter(Z[:,0],Z[:,1])
for i in range(len(index_word_map)):
	plt.annotate(s=index_word_map[i],xy=(Z[i,0],Z[i,1]))
plt.show()


#create 3d plot
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(Z[:,0],Z[:,1],Z[:,2],c='r',marker='o')

#uncomment to add annotations
# for i in range(len(index_word_map)):
#     ax.text(Z[i,0],Z[i,1],Z[i,2], index_word_map[i], zdir='z')

plt.show()








