"""
	Preprocessing utilities
"""
import numpy as np
import emoticons_ES
import twokenize_ES



def load_corpus(train_path,test_path):

	# Load data Training
	print("Loading Training data...")
	tweets, target = load_data(train_path)
	trainingPartitionLength = len(target)
	# Load data Testing
	print("Loading Testing data...")
	tweets_test, target_test = load_data(test_path)

	all_tweets = tweets+tweets_test
	all_target = target+target_test
	
	
	x, y, vocabulary = format_data(all_tweets,all_target)

	return [x, y, vocabulary, trainingPartitionLength]

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

	return [tweets,target]

def format_data(tw,tar):

	#Calculate maxLength
	maxLength = calculateMaxLength(tw)
	#Padding tweets...
	padded_tweets = padding(tw, maxLength)

	#Building vocabulary
	vocabulary = build_vocabulary(padded_tweets)

	#Mapping words with values in the vocabulary
	x = np.array([[vocabulary[word] for word in t] for t in padded_tweets])

	y = np.array(tar)

	return [x,y,vocabulary]

def padding(sentences, maxLength):
	"""
		Force all the sentences to be of the same length by filling with <PAD/>
	"""
	
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

def calculateMaxLength(sentences):
	return max(len(s) for s in sentences)


