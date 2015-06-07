#!/usr/bin/python

def readFile(prefix, D):
    with open(prefix+'expterm_wid','r') as idFile, open(prefix+'expterm','r') as wordFile:
        for (i,w) in zip(idFile, wordFile):
            D[ int(i.split(' ')[0]) ] = w.split(' ')[0]

def readDoc(docNum, D):
    docPrefix = "photo_system_corpus/{0}/".format(docNum)
    with open(docPrefix+'isAnno', 'r') as f:
        if(int(f.read()) == 1):
            readFile(docPrefix, D)

if __name__ == "__main__":
    D={}
    for i in xrange(7777):
        readDoc(i+1, D)
    with open('wordMap', 'w') as f:
        for key,value in sorted(D.iteritems()):
            f.write("{0} {1}\n".format(key, value))
