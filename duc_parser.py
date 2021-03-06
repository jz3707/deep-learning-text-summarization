#!/usr/bin/env python

# Yohanes Gultom (yohanes.gultom@gmail.com)
# Parse DUC dataset for summarization and generate feature matrix

from lxml import html
from os import listdir
from os.path import isfile, join
import sys
import math
import time

# return True if an int
def isint(x):
    try:
       val = int(x)
       return True
    except ValueError:
       return False

# normalize to 0..1
def normalize(x, min, max):
    return (x - min) / (max-min)

# print csv
def printcsv(arr):
    for row in arr:
        print ', '.join(['{:.2f}'.format(x) for x in row])

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
                            'stem' : features[3].lower(),
                            'pos' : features[4],
                            'tf' : float(features[5])
                        })
            # include only sentence with at least one word
            if (len(sentence['words']) > 0):
                res.append(sentence)
    return {
        'id' : file,
        'title' : res[0]['text'],
        'sentences' : res
    }

# parse all DUC documents inside a dir
def parse_duc_dir(dir):
    docs = []
    for f in listdir(dir):
        filepath = join(dir, f)
        if isfile(filepath):
            doc = parse_duc(filepath)
            docs.append(doc)
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
    return math.log(docLen / float(docTermLen))

# concept feature of two sequential terms in a doc
def concept_feature_two_terms(term1, term2, doc):
    termCompound = term1 + ' ' + term2
    total = 0
    appearance = {}
    appearance[term1] = 0
    appearance[term2] = 0
    appearance[termCompound] = 0
    # window = sentence
    total = len(doc['sentences'])
    for s in doc['sentences']:
        prevWord = None
        found = {}
        found[term1] = False
        found[term2] = False
        found[termCompound] = False
        for w in s['words']:
            currentWord = w['stem']
            # check appearance
            if currentWord.lower() == term1.lower():
                found[term1] = True
            elif currentWord.lower() == term2.lower():
                found[term2] = True
            # check co-appearance with previous word
            if prevWord != None:
                if (prevWord.lower() == term1.lower() and currentWord.lower() == term2.lower()):
                    found[termCompound] = True
            if currentWord != None: prevWord = currentWord
        if found[term1]:
            appearance[term1] += 1
        if found[term2]:
            appearance[term2] += 1
        if found[termCompound]:
            appearance[termCompound] += 1
    # calculate probability
    prob = {}
    prob[term1] = appearance[term1] / float(total)
    prob[term2] = appearance[term2] / float(total)
    prob[termCompound] = appearance[termCompound] / float(total)
    if (prob[term1] * prob[term2] == 0): print '{0} {1}'.format(term1, term2)
    wi = 2 * prob[termCompound] / (prob[term1] * prob[term2])
    return math.log(wi) if (wi > 0.0) else 0.0

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
    if (docLen == 1):
        pos = 1
    elif (sentencePos <= mean):
        # print 'sentencePos: {0} docLen: {1} mean: {2}'.format(sentencePos, docLen, mean)
        pos = (mean - sentencePos) / float(mean - 1)
    else:
        pos = (sentencePos - mean) / float(mean - 1)
    # print 's: {0} | doc: {1} | mean: {2} | pos: {3}'.format(sentencePos, docLen, mean, pos)
    return pos

# calculate f3 term weight = tf * idf
def term_weight(sentence, docs):
    tw = 0
    # try to use max instead of sum
    # just felt more reasonable (and easier :p)
    for w in sentence['words']:
        # tw = tw + w['tf'] * idf(w['stem'], docs)
        tempTw = w['tf'] * idf(w['stem'], docs)
        if (tempTw > tw):
            tw = tempTw
    return tw

# calculate f4 concept feature
def concept_feature(sentence, doc, cache = []):
    prevWord = None
    totalConceptFeature = 0
    totalCompound = 0
    # print sentence['text']
    if len(sentence['words']) > 1:
        for w in sentence['words']:
            currentWord = w['stem']
            #print 'prev: {0} | current: {1}'.format(prevWord, currentWord)
            if prevWord != None:
                # print 'd: {0} s: {1}'.format(doc['id'], sentence['id'])
                totalConceptFeature += concept_feature_two_terms(prevWord, currentWord, doc)
                totalCompound += 1
            if currentWord != None: prevWord = currentWord
    return (totalConceptFeature / float(totalCompound)) if (totalCompound > 0) else 0

# get feature matrices from directory containing documents under one topic. One matrix represent a docuement
def get_feature_matrix(dir, summaryDocs = None, printSentences = False):
    docs = parse_duc_dir(dir)
    feature_matrix = []
    f4max = 0
    f4min = 0
    for doc in docs:
        # print 'Title: {0}'.format(doc['title'])
        title = doc['sentences'][0]
        docLen = len(doc['sentences'])
        # start from 1 to skip title
        for i in range(1, docLen):
            s = doc['sentences'][i]
            # print s['text']
            f1 = title_similarity(title, s)
            f2 = positional_feature(i, docLen-1) # -1 because title is excluded
            f3 = term_weight(s, docs)
            f4 = concept_feature(s, doc)
            if (f4 > f4max): f4max = f4
            if (f4 < f4min): f4min = f4
            # print 'similarity: {0} | positional: {1} | term weight: {2} | concept: {3}'.format(f1, f2, f3, f4)
            # check summary sample
            if (printSentences):
                print s['text']
            elif (summaryDocs != None):
                summary = 0.0
                for sum in summaryDocs:
                    for ssum in sum['sentences']:
                        if ssum.lower() == s['text'].lower():
                            summary = 1.0
                            break
                    if (summary == 1.0): break
                feature_matrix.append([f1, f2, f3, f4, summary])
            else:
                feature_matrix.append([f1, f2, f3, f4])
            # print '{0}\t{1}\t{2}\t{3}'.format(f1, f2, f3, f4)
    # print 'min: {0} max: {1}'.format(f4min, f4max)
    # normalize f4
    for row in feature_matrix:
        row[3] = normalize(row[3], f4min, f4max)

    return feature_matrix


# read training sample
def get_summaries(dir, startWith):
    res = []
    # print 'Reading from: {0}'.format(filePath)
    for file in listdir(dir):
        if file.lower().startswith(startWith.lower()):
            sentences = []
            filepath = join(dir, file)
            with open(filepath, 'r') as f:
                for line in f:
                    sentences.append(line.strip())
            f.closed
            res.append({
                'id' : filepath,
                'sentences' : sentences
            })
    return res

# croscheck summary and original docs
def describe_summaries(summaryDir, summaryStartWith, originalDir):
    docs = parse_duc_dir(originalDir)
    summaries = get_summaries(summaryDir, summaryStartWith)
    print 'docs: {0} summary: {1}'.format(len(docs), len(summaries))

    for sum in summaries:
        print '=============> {0}'.format(sum['id'])
        for ssum in sum['sentences']:
            found = False
            for doc in docs:
                for s in doc['sentences']:
                    if (s['text'].lower() == ssum.lower()):
                        print '{0}: {1}'.format(doc['id'], s['id'])
                        found = True
                        break
                if (found): break
            if (found): break
            if (not found): print 'notfound'


# main program
# python duc_parser.py /home/yohanes/Workspace/duc/06/D0601A /home/yohanes/Workspace/duc/original/06/models D0601
# python duc_parser.py /home/yohanes/Workspace/duc/06/D0601A > D0601.txt
if __name__ == "__main__":
    # ticks = time.time() * -1
    summaries = None
    argc = len(sys.argv)
    if argc <= 2:
        feature_matrix = get_feature_matrix(sys.argv[1], None, True)
    elif argc >= 4:
        summaries = get_summaries(sys.argv[2], sys.argv[3])
        feature_matrix = get_feature_matrix(sys.argv[1], summaries)
        printcsv(feature_matrix)
    # if argc > 2:
    #     feature_matrix = get_feature_matrix(sys.argv[1], summaries)
    #     printcsv(feature_matrix)
    #     duration = (time.time() + ticks)
    #     # print 'done. time elapsed {0} s'.format(duration)

    # describe_summaries(sys.argv[2], sys.argv[3], sys.argv[1])
    # print get_feature_matrix(sys.argv[1])
