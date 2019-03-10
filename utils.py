import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia 

import csv 
import numpy as np
import pandas as pd 
from datetime import datetime
import random
import collections
import re
import string 
import datetime

def merge_rows(data): 
    init = []
    for ele1 in data: 
        for ele2 in ele1: 
            init.append(ele2)
    return init


def delstring(orig, string):
    strings = re.findall(string, orig)
    outp=orig
    for i in strings:
        outp = re.sub(i, '', orig)
        
    return outp

def replacestring(orig, string1, string2):
    strings = re.findall(string1, orig)
    outp=orig
    for i in strings:
        outp = re.sub(i, string2, orig)
        
    return outp

def gethashs(tweets):
    
    hashs = []
    
    for i in tweets:
        hash_one = re.findall(r"#(\w+)", i)
        hashs.append(hash_one)

    return hashs

def getats(tweets):
    
    ats = []
    
    for i in tweets:
        at_one = re.findall(r"@(\w+)", i)
        ats.append(at_one)

    return ats

def getlinks(tweets):
    
    links = []
    
    for i in tweets:
        temp = []
        for word in i.split(' '): 
            if word.startswith('https:'): 
                temp.append(word)
        links.append(temp)
       
    return links


def getchildren(tweets):
    
    children = []
    names = ['eric', 'trumpjr', 'tiffany','ivanka', 'barron']
    for i in tweets:
        temp = []
        for word in i.split(' '): 
            for name in names: 
                if name in word.lower():   
                    temp.append(name)
        children.append(temp)
       
    return children

def feature_engineering(data, power=1): 
    
    data['Dates'] = pd.to_datetime(data['created']).dt.date
    data['Time'] = pd.to_datetime(data['created']).dt.time
    data['hour'] = data['Time'].astype('str').str.split(':').apply (lambda x: int(x[0]) + int(x[1])/60.0)
    data['newdate'] = pd.to_datetime(data['Dates']).dt.strftime('%y%j')

    data['hashs'] = gethashs(data['text'])
    data['links'] = getlinks(data['text'])
    data['ats'] = getats(data['text'])
    data['children'] =  getchildren(data['text'])
    data['nhashs'] = [len(x) for x in data['hashs']]
    data['nlinks'] = [len(x) for x in data['links']]
    data['nats'] = [len(x) for x in data['ats']]
    data['nchildren'] = [len(x) for x in data['children']]

    data['cleantext'] = data['text'].apply(lambda x: ' '.join([w for w in x.split() if ("https:" not in w and "@" not in w and "#" not in w)]))
    data['cleantext'] = data['cleantext'].apply(lambda x: ' '.join([w for w in x.split() if ( "&amp" not in w)]))
    data['cleantext'] = data['cleantext'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
    data['ncaps'] = data['cleantext'].apply(lambda x: len([w for w in x.split() if w.isupper()]))
    data['length'] = data['text'].apply(lambda x: len(x.split()))
    data['nself'] = data['text'].apply(lambda x: len([w for w in x.split() if 'realDonaldTrump' in w ]))
    data['ncampaign'] = data['text'].apply(lambda x: len([w for w in x.split() if 'Trump2016' in w ]))
    data['rt'] = data['text'].apply(lambda x : len([w for w in x.split() if '\"@' in w ]))
    
    sia_init=sia()
    
    data['pos'] = data['cleantext'].apply(lambda x: sia_init.polarity_scores(x)['pos']**power)
    data['neg'] = data['cleantext'].apply(lambda x: sia_init.polarity_scores(x)['neg']**power)
    data['neu'] = data['cleantext'].apply(lambda x: sia_init.polarity_scores(x)['neu']**power)
    
    data['cleantext'] = data['cleantext'].str.replace("[^a-zA-Z#]", " ")
    data['cleantext'] = data['cleantext'].apply(lambda x: ' '.join([w.lower() for w in x.split()]))

    return data 