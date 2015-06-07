#!/usr/bin/python

def readDoc(docNum):
    docPrefix = "photo_system_corpus/{0}/".format(docNum)
    maxIdx = 0
    with open(docPrefix+'isAnno', 'r') as f:
        if(int(f.read()) == 1):    
            with open(docPrefix+'expterm_wid', 'r') as f:
                for line in f:
                    [idx, count]=line.split()
                    idx = int(idx)
                    if(idx > maxIdx):
                        maxIdx = idx
    print("Doc{0} max:{1}".format(docNum, maxIdx))
    return maxIdx
    
if __name__ == "__main__":
    maxNum = 0
    idx = 0
    for i in xrange(7777):
        num = readDoc(i+1)
        if(num > maxNum):
            maxNum = num
            idx = i
    print("the max speech term is {0}.({1})".format(maxNum,idx))
