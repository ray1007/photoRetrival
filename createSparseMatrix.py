#!/usr/bin/python

from sys import argv

import numpy as np
from numpy import linalg as LA

import const

'''
numSpeech = 51135
numViWord = 10000
numGVWord = 374
numFeat   = numSpeech + numViWord + numGVWord
numPhoto  = 7777
'''

annotated_photo_idx = []

def readDoc(docNum, A):
    docPrefix = "photo_system_corpus/{0}/".format(docNum)
    with open(docPrefix+'isAnno', 'r') as f:
        if(int(f.read()) == 1):
            annotated_photo_idx.append(docNum-1)
            with open(docPrefix+'expterm_wid', 'r') as f:
                for line in f:
                    [speechID, wc] = line.split(' ')
                    [speechID, wc] = [int(speechID), float(wc)]
                    A[docNum-1, speechID-1] = wc
    with open(docPrefix+'visual_word', 'r') as f:
        for line in f:
            [vwID, wc] = line.split(' ')
            [vwID, wc] = [int(vwID), float(wc)]
            A[docNum-1, const.NUM_SPEECH+vwID-1] = wc
    with open(docPrefix+'col374', 'r') as f:
        for line in f:
            [gvwID, wc] = line.split(' ')
            [gvwID, wc] = [int(gvwID), float(wc)]
            A[docNum-1, const.NUM_SPEECH+const.NUM_VIWORD+gvwID-1] = wc
            #print("A[{0},{1}] = {2}".format(docNum,numSpeech+vwID,wc))
            
def createMatrix(filename):
    dim=(const.NUM_PHOTO, const.NUM_FEAT)
    A = np.memmap(filename, dtype='float32', mode='w+', shape=dim)
    #A.fill(float(1e-6))
    for i in xrange(1, const.NUM_PHOTO+1):
        if(i%1000 == 0):
            print("Parsed photo {0}".format(i+1))
        readDoc(i, A)
    return A

def enhance(A):
    A_enh = np.memmap('nmf_a_enhanced',dtype='float32',mode='w+',shape=A.shape)
    A_enh[:] = A[:]
    begin_idx = const.NUM_SPEECH + const.NUM_VIWORD
    end_idx = begin_idx + const.NUM_GVWORD
    a = 0.3
    for i in annotated_photo_idx:
        print i
        i_gv = A[i,begin_idx:end_idx]
        for j in xrange(const.NUM_PHOTO):
            j_gv = A[j,begin_idx:end_idx]
            sim = 0
            if LA.norm(i_gv)!=0 and LA.norm(j_gv)!=0:
                sim = np.dot(i_gv, j_gv) / (LA.norm(i_gv)*LA.norm(j_gv))
            if sim > 0:
                A_enh[j,0:const.NUM_SPEECH] += a * sim * A[i,0:const.NUM_SPEECH]

if __name__ == "__main__":
    # Read in all corpus data and create matrix.
    filename = 'nmf_a'
    if len(argv) == 2:
        filename = argv[1]
    A = createMatrix(filename)
    print("Created A.")
    enhance(A)
    print("Enhanced A.")
