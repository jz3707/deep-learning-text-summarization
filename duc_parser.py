#!/usr/bin/env python

from lxml import html

def isint(x):
    try:
       val = int(x)
       return True
    except ValueError:
       return False

def parse_duc(file):
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
                    sentence['words'].append({
                        'text' : features[0],
                        'stop' : int(features[1]) == 1,
                        'offset' : features[2],
                        'stem' : features[3],
                        'pos' : features[4],
                        'tf' : float(features[5])
                    })
            res.append(sentence)
    return res


doc = parse_duc('/home/yohanes/Workspace/duc/05/d301i/FT921-10162.prep')
for s in doc:
    print 'id: {0} text: {1}'.format(s['id'], s['text'])
    for w in s['words']:
        print 'text: {0} stop: {1} offset: {2} stem: {3} pos: {4} tf: {5}'.format(w['text'], w['stop'], w['offset'], w['stem'], w['pos'], w['tf'])
