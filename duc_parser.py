#!/usr/bin/env python

from lxml import html
from os import listdir
from os.path import isfile, join
import sys
import math

# return True if an int
def isint(x):
    try:
       val = int(x)
       return True
    except ValueError:
       return False

# parse a DUC document
def parse_duc(file, removeStop = True):
    doc = html.parse(file)
    res = []
    for el in doc.getroot()[0]:
        if el.tag == 'sentence' and isint(el.attrib['id']):
            sentence = {
                'id' : int(el.attrib['id']),
                'text' : el.text.strip(),
                'words' : []
            }
            words = el[0].text.split('\n')
            for w in words:
                features = w.split('\t')
                if (len(features) == 6):
                    # remove stop words
                    if ((int(features[1]) == 1) or (removeStop == False)):
                        sentence['words'].append({
                            'text' : features[0],
                            'stop' : int(features[1]) == -1,
                            'offset' : features[2],
                            'stem' : features[3],
                            'pos' : features[4],
                            'tf' : float(features[5])
                        })
            # include only sentence with at least one word
            if (len(sentence['words']) > 0):
                res.append(sentence)
    return res

# parse all DUC documents inside a dir
def parse_duc_dir(dir):
    docs = []
    for f in listdir(dir):
        filepath = join(dir, f)
        if isfile(filepath):
            sentences = parse_duc(filepath)
            docs.append({
                'id' : f,
                'title' : sentences[0]['text'],
                'sentences' : sentences
            })
    return docs

# count idf of a term = log (total document / total document containing the term)
def idf(term, docs):
    docLen = len(docs)
    docTermLen = 0
    for doc in docs:
        found = False
        for s in doc['sentences']:
            for w in s['words']:
                if term.lower() == w['stem'].lower():
                    found = True
                    break
            if found: break
        if found: docTermLen += 1
    return math.log10(docLen / float(docTermLen))

# return true if a term exist in a sentence
def contains_term(term, sentence):
    for w in sentence['words']:
        if (w['stem'].lower() == term.lower()):
            return True
    return False

# concept feature of two sequential terms in a doc
def concept_feature_two_terms(term1, term2, doc):
    total = 0
    totalCompound = 0
    appearance[term1] = 0
    appearance[term2] = 0
    appearance[term1 + ' ' + term2] = 0
    prevWord = None
    for s in doc['sentences']:
        for w in s['words']:
            total +=1
            currentWord = w['stem']
            # check appearance
            if currentWord.lower() == term1.lower():
                appearance[term1] += 1
            elif currentWord.lower() == term2.lower():
                appearance[term2] += 1
            # check co-appearance with previous word
            if prevTerm == None:
                prevWord = currentWord
            elif (prevWord.lower() == term1.lower() and currentWord.lower() == term2.lower()):
                appearance[term1 + ' ' + term2] += 1
    # calculate probability
    appearance[term1] = appearance[term1] / float(total)
    appearance[term2] = appearance[term2] / float(total)
    appearance[term1 + ' ' + term2] = appearance[term1 + ' ' + term2] / float(totalCompound)
    return math.log(appearance[term1 + ' ' + term2] / appearance[term1] * appearance[term2], 2)

# calculate f1 title similarity
def title_similarity(title, sentence):
    similar = 0
    sentenceLen = len(sentence['words'])
    if (sentenceLen > 0):
        for ws in sentence['words']:
            for wt in title['words']:
                similar = similar + 1 if (ws['stem'].lower() == wt['stem'].lower()) else similar
        similar = similar / float(sentenceLen)
    return similar

# calculate f2 positional feature
def positional_feature(sentencePos, docLen):
    mean = (docLen + 1) / float(2)
    if (sentencePos <= mean):
        pos = (mean - sentencePos) / float(mean - 1)
    else:
        pos = (sentencePos - mean) / float(mean - 1)
    # print 's: {0} | doc: {1} | mean: {2} | pos: {3}'.format(sentencePos, docLen, mean, pos)
    return pos

# calculate f3 term weight = tf * idf
def term_weight(sentence, docs):
    sentenceLen = len(sentence['words'])
    tw = 0
    for w in sentence['words']:
        tw = tw + w['tf'] * idf(w['stem'], docs)
    return tw / sentenceLen

# calculate f4 concept feature
def concept_feature():

    return 0

# main program
# expected sys.argv[1] = /home/yohanes/Workspace/duc/05/d301i
if (len(sys.argv) >= 2):
    dir = sys.argv[1]
else:
    sys.exit('please provide dir path of DUC dataset')
docs = parse_duc_dir(dir)
for doc in docs:
    print 'Title: {0}'.format(doc['title'])
    title = doc['sentences'][0]
    docLen = len(doc['sentences'])
    # start from 1 to skip title
    for i in range(1, docLen):
        s = doc['sentences'][i]
        print s['text']
        f1 = title_similarity(title, s)
        f2 = positional_feature(i, docLen-1) # -1 because title is excluded
        f3 = term_weight(s, docs)
        print 'similarity: {0} | positional: {1} | term weight: {2}'.format(f1, f2, f3)
