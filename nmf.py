#!/usr/bin/python

import numpy as np
from time import time
import pdb
import gc as gc

#np.seterr(divide='print')
print("gc.isenabled()={0}".format(gc.isenabled()))

def euclideanDist(A, B):
    if A.shape != B.shape:
        return None
    return np.sqrt(np.memmap.sum((A-B)**2))

def KLDivergence(A, B):
    if A.shape != B.shape:
        return None
    return np.memmap.sum(A*np.memmap.log(A/B)-A+B)

costFuncs = {
    'euclidean' : euclideanDist,
    'klDiv' : KLDivergence
}

def euclideanUpdate(V,W,H):
    Wtemp = np.memmap.copy(W)
    Htemp = np.memmap.copy(H)
    WH=np.memmap.dot(W,H)
    Htemp *= (np.memmap.dot(W.T,V)+float(1e-8))
    Htemp /= (np.memmap.dot(W.T,WH)+float(1e-8))

    WH=np.memmap.dot(W,Htemp)
    Wtemp *= (np.memmap.dot(V, H.T)+float(1e-8))
    Wtemp /= (np.memmap.dot(WH,H.T)+float(1e-8))
    return (Wtemp, Htemp)

def KLDivUpdate(V,W,H):
    Wtemp = np.copy(W)
    Htemp = np.copy(H)
    WH=np.dot(W,H)
    for (a,u),h in np.ndenumerate(H): # update H.
        numer=0.0;
        denom=0.0;
        for i in xrange(W.shape[0]):
            numer += float(W[(i,a)]) * V[(i,u)] / WH[(i,u)]
            denom += float(W[(i,a)])
        Htemp[(a,u)] = float(Htemp[(a,u)])*numer/denom	
    WH=np.dot(W,Htemp)
    for (i,a),w in np.ndenumerate(W): # update W.
        numer=0.0;
        denom=0.0;
        for u in xrange(H.shape[1]):
            numer += float(H[(a,u)]) * V[(i,u)] / WH[(i,u)]
            denom += float(H[(a,u)])
        Wtemp[(i,a)] = float(w)*numer/denom
    return (Wtemp, Htemp)

updateFuncs = {
    'euclidean' : euclideanUpdate,
    'klDiv' : KLDivUpdate
}

class nmf_config:
    def __init__(self, r, minDelta, costFuncName):
        self.r = r
        self.minDelta   = minDelta
        self.costFuncName = costFuncName
        self.costFunc = costFuncs[costFuncName]
        self.update = updateFuncs[costFuncName]

def nmf(V, config):
    """
    Non-negative Matrix Factorization
    
    returns : (W,H), where W is a 
    """
    # calculate execution time.
    print("In nmf.py.")
    beginTime = time()

    # Initialize W & H.
    dimW = (V.shape[0], config.r)
    dimH = (config.r, V.shape[1])
    #W = np.random.uniform(0, 1, dimW)
    #H = np.random.uniform(0, 1, dimH)
    #WH= np.dot(W,H)
    W = np.memmap('W.tmp', dtype='float64', mode='w+', shape=dimW)
    H = np.memmap('H.tmp', dtype='float64', mode='w+', shape=dimH)
    WH = np.memmap.dot(W,H)
    print("WH calculated.")
    WH_old = None
    #pdb.set_trace()
    currCost = config.costFunc(V,WH)
    lastCost = currCost
    iterCount = 0
    print("Init cost:{0}".format(currCost))
	# Run multiplicative update rule.
    for it in xrange(1000):
        Wtemp, Htemp = config.update(V,W,H)
        WH_old = WH
        WH=np.memmap.dot(Wtemp, Htemp)
        # update cost.
        temp = config.costFunc(V,WH)
        #print("lastcost:{0}, currCost:{1}, temp:{2}".format(lastCost,currCost,temp))
        #print("differnce between WH: {0}".format(config.costFunc(WH_old,WH)))
        if(lastCost != currCost and lastCost - temp < config.minDelta):
            print("break since {0} -> {1}".format(lastCost,temp))
            break
        W = np.memmap.copy(Wtemp)
        H = np.memmap.copy(Htemp)
        lastCost = currCost
        currCost = temp
        iterCount+=1
        #if(iterCount%10 == 0):
        #	print("Iter = {0}".format(iterCount))

    print("Done nmf algorithm.")
    print(" (n,m,r)=({0},{1},{2})".format(V.shape[0], V.shape[1], config.r))
    print(" costFunc="+config.costFuncName)
    print(" cost={0}->{1}".format(lastCost,currCost))
    print(" minDelta={0}".format(config.minDelta))
    print(" iterations={0:d}".format(iterCount))
    print(" time eplased={0:.0f}".format(time()-beginTime))
    #pdb.set_trace()
    return (W,H)

if __name__ == "__main__":
    config = nmf_config(3, 0.0000001, "euclidean")
    #config = nmf_config(3, 0.0000001, "klDiv")
    V = np.random.uniform(100,200,(10, 7))
    print("start nmf.")
    (W,H) = nmf(V,config)
    with open('result.txt','w') as file:
        WH = np.dot(W,H)
        for idx, v in np.ndenumerate(V):
            file.write("{0} {1}\n".format(V[idx], WH[idx]))
