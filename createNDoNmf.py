#!/usr/bin/python

from sys import argv
import numpy as np
import nmf as nmf
from scipy.sparse import coo_matrix 

numSpeech = 51135
numViWord = 10000
numGVWord = 374
numFeat   = numSpeech + numViWord + numGVWord
numPhoto  = 7777

def readDoc(docNum, A):
    docPrefix = "photo_system_corpus/{0}/".format(docNum)
    with open(docPrefix+'isAnno', 'r') as f:
        if(int(f.read()) == 1):    
            with open(docPrefix+'expterm_wid', 'r') as f:
                for line in f:
                    [speechID, wc] = line.split(' ')
                    [speechID, wc] = [int(speechID), float(wc)]
                    A[docNum-1, speechID-1] = wc
    with open(docPrefix+'visual_word', 'r') as f:
        for line in f:
            [vwID, wc] = line.split(' ')
            [vwID, wc] = [int(vwID), float(wc)]
            A[docNum-1, numSpeech+vwID-1] = wc
    with open(docPrefix+'col374', 'r') as f:
        for line in f:
            [gvwID, wc] = line.split(' ')
            [gvwID, wc] = [int(gvwID), float(wc)]
            A[docNum-1, numSpeech+numViWord+gvwID-1] = wc
            #print("A[{0},{1}] = {2}".format(docNum,numSpeech+vwID,wc))
            
def createMatrix():
    #A = np.zeros(shape=(numPhoto, numFeat), dtype=np.float16)
    #A = coo_matrix((numPhoto, numFeat), dtype=np.float)
    A = np.memmap('nmf_a', dtype='float64', mode='w+', shape=(numPhoto,numFeat))
    for i in xrange(1, numPhoto+1):
        if(i%1000 == 0):
            print("Parsed photo {0}".format(i+1))
        readDoc(i, A)
    return A

if __name__ == "__main__":
    # Read in all corpus data and create matrix.
    A = createMatrix()
    #np.savetxt('nmf_A.txt',A)
    print("Created A.")
    # Start nmf.
    d = 90
    config = nmf.nmf_config(d, 0.00001, 'euclidean')
    (W,H) = nmf.nmf(A,config)
    np.savetxt('nmf_W.txt',W)
    np.savetxt('nmf_H.txt',H) 
