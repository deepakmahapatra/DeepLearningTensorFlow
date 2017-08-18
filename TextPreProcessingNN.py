
# coding: utf-8

# In[17]:

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter


# In[32]:

import sys
reload(sys)
sys.setdefaultencoding("utf-8")


# In[33]:

lemmatizer=WordNetLemmatizer()
num_lines=100000




# In[34]:

def lemmatize(x):
    return lemmatizer.lemmatize(x)


# In[51]:

def lexiconCreation(positive, negative):
    lexicon=[]
    word_counts=Counter()
    for fi in [positive,negative]:
        with open (fi,'r') as f:
            contents=f.readlines()
            for line in contents[:num_lines]:
                this_set_words=word_tokenize(line.lower())
                lexicon=list(this_set_words)
                lexicon=map(lemmatizer.lemmatize,lexicon)
                #[lemmatizer.lemmatize(i) for i in lexicon]
                temp_counter=Counter(lexicon)
                word_counts=(word_counts+temp_counter)
    lexicon=[]
    for word_cnt in word_counts:
        if 1000>word_counts[word_cnt]>50:
            lexicon.append(word_cnt)
    return lexicon
                
                
                
        


# In[52]:

def sample_handling(sample, lexicon, classification):
    featureset=[]
    with open(sample,'r') as f:
        contents=f.readlines()
        for l in contents[:num_lines]:
            current_words=word_tokenize(l.lower())
            current_words=map(lemmatizer.lemmatize,current_words)
            #for i in current_words]
            features=np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value=lexicon.index(word.lower())
                    features[index_value]+=1
            features=list(features)
            featureset.append([features, classification])
    return featureset
    


# In[53]:

def create_feature_sets_and_labels(pos,neg,test_size=0.2):
    lexicon=lexiconCreation(pos,neg)
    features=[]
    features+=sample_handling('pos.txt',lexicon,[1,0])
    features+=sample_handling('neg.txt',lexicon,[0,1])
    random.shuffle(features)
    
    features=np.array(features)
    
    testing_size=int(test_size*len(features))
    
    train_x=list(features[:,0][:-testing_size])
    train_y=list(features[:,1][:-testing_size])
    test_x=list(features[:,0][-testing_size:])
    test_y=list(features[:,1][-testing_size:])
    
    return train_x,train_y,test_x,test_y
    
    
    


# In[ ]:

if __name__=='__main__':
    train_x,train_y,test_x,test_y=create_feature_sets_and_labels('pos.txt','neg.txt')


# In[ ]:

train_x


# In[ ]:




# In[ ]:



