#From 2018 SemEval competition:
#Valence ordinal classification subtask 4
#Mackenna Barker
#
#
#
# This is an ordinal classification task that takes a set of tweets
# scored by valence, or intensity, of emotion exhibited and learns how
# to classify a sentence on a 7 point scale. Model was trained and tested
# using the data provided from the 2018 SemEval competition for Task 1:
# V-oc which can be found here:
# https://competitions.codalab.org/competitions/17751#learn_the_details-datasets
#
#


import numpy as np
import csv
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, casual
from nltk.tokenize.casual import TweetTokenizer
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# preprocessing steps
stop_words = set(stopwords.words('english'))
#adding additional stop words that won't be useful in classification 
stop_words.add('#')
stop_words.add('@')
stop_words.add(',')
stop_words.add('.')
stop_words.add('-')
stop_words.add(":")
stop_words.add(";")

def emojiChecker(s):
    # checks if tweet/string has emoji
    # stores emoji separately 
    array =[]
    for i in range(0,len(s)):
        # no emojis
        if ord(s[i]) < 128:
            pass
        else:
            array.append(s[i])
            return array
    return 0
    
def removeNonAscii(s): 
    # removes non-ascii characters such as emojis and foreign text
    return "".join(i for i in s if ord(i)<128)

def get_bigrams(s):
    # gets all possible bigrams as a list of tuples from a given tweet
    if s != None:
        emoji_array = emojiChecker(s)
        if emoji_array == 0:
            word_tokens = word_tokenize(s)
            filtered_sent = [w for w in word_tokens if not w in stop_words]
            blist = list(nltk.bigrams(filtered_sent))
        else:
            no_emoji = removeNonAscii(s)
            word_tokens = word_tokenize(no_emoji)
            filtered_sent = [w for w in word_tokens if not w in stop_words]
            blist = list(nltk.bigrams(filtered_sent))
        return blist
    else:
        return None
        
samples = []
with open('2018-Valence-oc-En-train.txt', encoding='utf8') as inputfile:
    for row in csv.reader(inputfile):
        samples.append(row)

#all of the tweets are separated by line now
parsed_samples = np.empty((len(samples), 4), dtype='object')
start_index = 0
end_index = 0
counter = 0

# reading from file
for i in range(0, len(samples)):
    for j in range(0, len(samples[i][0])):
        if samples[i][0][j] == '\t':
            end_index = j
            #print(samples[i][0][start_index:end_index])
            parsed_samples[i][counter] = samples[i][0][start_index:end_index]
            counter += 1
            start_index = end_index + 1
        if counter == 3:
            parsed_samples[i][counter] = samples[i][0][start_index:len(samples[i][0])]
    counter = 0
    start_index = 0

# getting tweet, bigrams (as an array of lists and as a single list), scores from text strings
tweets, bigrams_list, all_bigrams, score_list = [],[],[],[]

for i in range(0, len(samples)):
    tweet = parsed_samples[i][1]
    if tweet != None:
        tweets.append(tweet) 
    blist = get_bigrams(tweet)
    if blist != None:
        bigrams_list.append(blist)
        for a in blist:
            if a not in all_bigrams:
                all_bigrams.append(a) 
    score = str(parsed_samples[i][3])
    aScore = score.partition(':')
    if aScore[0] != 'None':
        score_list.append(int(aScore[0]))
        #print aScore[0]

# vectorize the bigrams, term frequency
vectorized = np.ndarray((len(score_list),len(all_bigrams)))
#current vector we'll be appending, one per tweet
current_vector = np.zeros(len(all_bigrams))

#set up term frequency values
for k in range(0,len(bigrams_list)):
    for l in range(0,len(bigrams_list[k])):
        add_index = all_bigrams.index(bigrams_list[k][l])
        current_vector[add_index] = 1
    vectorized[k] = current_vector
    current_vector = np.zeros(len(all_bigrams))

#set up classifier
clf = svm.SVR()

#split up train and test portions
X_train, X_test, y_train, y_test = train_test_split(vectorized, score_list, test_size=0.1, random_state=42)
#fit our classifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
clf.fit(X_train,y_train) 
#predict on our test samples
pred = clf.predict(X_test)  

#evaluation
print ("true labels:", y_test)
print ("predicated labels:", pred)
print (classification_report(y_test, pred, target_names=['-3', '-2', '-1', '0','1','2','3']))

"""
output:
true labels: [0, 2, -1, -3, 1, 3, 0, -3, 0, 0, -2, 3, -2, -2, -2, 0, -3, 0, -3, -2, 1, -2, -3, -2, 0, -3, 0, 1, -3, 0, -1, 0, 1, 1, -3, 0, -2, 0, 0, 2, 1, -2, -2, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 3, -3, -3, -2, 2, 0, -2, 0, 0, -2, 0, 0, 0, 0, 0, -3, -1, -2, -2, 1, 0, 0, -2, 2, -1, 0, 0, -2, -3, -1, 1, 1, 0, -1, 0]
predicated labels: [ 0  1  0  0  2  0  0  0  0  0 -2  3  0 -3  0  0  0  0  0  0  3  0 -3  0  0
  0  0  0  0  0 -3  0 -3 -1  0  0  0  3  0  3  0  0  0 -3  0  3 -2  0 -1  0
  0  0  0  0  0  0  0  0  1  1  0  0  0  0  0  0  0 -2  1  0  3  3  0  0  0
 -3  0  0  0 -3  0 -3  0  0  0  0  0  0]
             precision    recall  f1-score   support

         -3       0.25      0.17      0.20        12
         -2       0.67      0.10      0.17        21
         -1       0.00      0.00      0.00         6
          0       0.43      0.82      0.56        33
          1       0.00      0.00      0.00         9
          2       0.00      0.00      0.00         4
          3       0.14      0.33      0.20         3

avg / total       0.36      0.36      0.28        88
"""
