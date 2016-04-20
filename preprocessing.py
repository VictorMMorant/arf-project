"""
	Preprocessing utilities
"""
import numpy as np
import emoticons_ES
import twokenize_ES


#Retrieve training data
def load_data(path):
	"""
		Load the tweets and labels from the data file in the path.
	"""

	f = open(path,'r');

	tweets = [];
	target = [];
	for line in f :
		if line != '' and line != '\n':
			listLine = line.strip().split('\t');
			
			#Tokenize tweet
			listLine[0] = u" ".join(twokenize_ES.tokenize(listLine[0]))
			
			#Analize tweet
			listLine[0] = emoticons_ES.analyze_tweet(listLine[0])
			
			#RemovePunctuation
			listLine[0] = u" ".join(twokenize_ES.remove_punct(listLine[0]))

			tweets.append(listLine[0].strip().split());
			if listLine[1] == 'positive':
				target.append([1,0,0])
			elif listLine[1] == 'negative':
				target.append([0,0,1])
			else:
				target.append([0,1,0])

	y = np.array(target)

	#Padding tweets...
	padded_tweets = padding(tweets)

	#Building vocabulary
	vocabulary = build_vocabulary(padded_tweets)

	#Mapping words with values in the vocabulary
	x = np.array([[vocabulary[word] for word in t] for t in padded_tweets])

	return [x, y, vocabulary]

def padding(sentences):
	"""
		Force all the sentences to be of the same length by filling with <PAD/>
	"""
	maxLength = max(len(s) for s in sentences)
	
	padded_sentences = []
	for i in range(len(sentences)):
		s = sentences[i]
		nPads = maxLength - len(s)
		new_s = s + ["<PAD/>"]*nPads
		padded_sentences.append(new_s)	

	return padded_sentences

def build_vocabulary(sentences):
	"""
		Build a mapping between every word and 
	"""
	words = []
	for sentence in sentences:
		for word in sentence:
			words.append(word)
	words = sorted(set(words))
	vocabulary = {x: i for i, x in enumerate(words)}

	return vocabulary

