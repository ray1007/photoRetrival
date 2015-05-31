#!/usr/bin/python

import numpy as np
import pdb

batch = 200

def memDot(outname, in1, in2):
    sh = (in1.shape[0], in2.shape[1])
    result = np.memmap(outname, dtype='float32', mode='r+', \
                       shape=sh)
    for i in xrange(sh[0]/batch):
        print("{0} of {1}".format(i,sh[0]/batch))
        s = i*batch
        e = (i+1)*batch
        if(e > sh[0]): e=sh[0]
        result[s:e,:] = np.dot(in1[s:e,:],in2)

def euclideanDist(Afilename, Bfilename, Vshape):
    A = np.memmap(Afilename, dtype='float32', mode='r', \
                  shape=Vshape)
    B = np.memmap(Bfilename, dtype='float32', mode='r', \
                  shape=Vshape)
    if A.shape != B.shape:
        return None
    distSum=0
    for i in xrange(A.shape[0]/batch):
        print("{0} of {1}".format(i,A.shape[0]/batch))
        s = i*batch
        e = (i+1)*batch
        if(e > A.shape[0]): e=A.shape[0]
        distSum += np.sum((A[s:e,:]-B[s:e,:])**2)
    return np.sqrt(distSum)


def euclideanUpdate(V,W,H):
    W_old = np.copy(W)
    H_old = np.copy(H)
    memDot('WH.tmp', W, H)
    WH=np.memmap('WH.tmp', dtype='float32', mode='r', \
                 shape=(W.shape[0], H.shape[1]))
    for i in xrange(H_old.shape[1]/batch):
        print("{0} of {1}".format(i,H_old.shape[1]/batch))
        s = i*batch
        e = (i+1)*batch
        if(e > H_old.shape[1]): e=H_old.shape[1]
        H[:,s:e] *= (np.dot(W_old.T, V[:,s:e])+float(1e-8))
        H[:,s:e] /= (np.dot(W_old.T, WH[:,s:e])+float(1e-8))

    memDot('WH.tmp', W, H)
    for i in xrange(W_old.shape[0]/batch):
        print("{0} of {1}".format(i,W_old.shape[0]/batch))
        s = i*batch
        e = (i+1)*batch
        if(e > W_old.shape[0]): e=W_old.shape[0]
        W[s:e,:] *= (np.dot(V[s:e,:], H_old.T)+float(1e-8))
        W[s:e,:] /= (np.dot(WH[s:e,:],H_old.T)+float(1e-8))

costFuncs = {
    'euclidean' : euclideanDist,
    #'klDiv' : KLDivergence
}
updateFuncs = {
    'euclidean' : euclideanUpdate,
    #'klDiv' : KLDivUpdate
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
    """
    print("In nmf.py.")

    # Initialize W & H.
    Vfname  = 'nmf_a'
    WHfname = 'WH.tmp'
    dimV = V.shape
    dimW = (V.shape[0], config.r)
    dimH = (config.r, V.shape[1])
    maxIter = 1000
    W = np.memmap('nmf_W', dtype='float32', mode='w+', shape=dimW)
    H = np.memmap('nmf_H', dtype='float32', mode='w+', shape=dimH)
    W.fill(1)
    H.fill(1)
    memDot('WH.tmp', W, H)
    print("Init WH calculated.")
    pdb.set_trace()
    currCost = config.costFunc(Vfname, WHfname, dimV)
    lastCost = currCost
    iterCount = 0
    print("Init cost:{0}".format(currCost))
	# Run multiplicative update rule.
    for it in xrange(maxIter):
        if it % 500 == 0: print(it)
        config.update(V,W,H)
        memDot(W, H)
        newCost = config.costFunc(Vfname, WHfname, dimV)
        print("lastcost:{0}, currCost:{1}, temp:{2}".format(lastCost,currCost,temp))
        if(lastCost != currCost and lastCost - temp < config.minDelta):
            print("break since {0} -> {1}".format(lastCost,temp))
            break
        lastCost = currCost
        currCost = newCost
        iterCount+=1

    print("Done nmf algorithm.")
    print(" (n,m,r)=({0},{1},{2})".format(V.shape[0], V.shape[1], config.r))
    print(" costFunc="+config.costFuncName)
    print(" cost={0}->{1}".format(lastCost,currCost))
    print(" minDelta={0}".format(config.minDelta))
    print(" iterations={0:d}".format(iterCount))
    #print(" time eplased={0:.0f}".format(time()-beginTime))

if __name__ == "__main__":
    config = nmf_config(90, 0.0000001, "euclidean")
    #config = nmf_config(3, 0.0000001, "klDiv")
    numFeat   = 61509
    numPhoto  = 7777
    V = np.memmap('nmf_a', dtype='float32', mode='r+', \
                  shape=(numPhoto, numFeat))
    print("start nmf.")
    nmf(V,config)

