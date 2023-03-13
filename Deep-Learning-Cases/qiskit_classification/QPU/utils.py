# Created by Dennis Willsch (d.willsch@fz-juelich.de) 
# Modified by Gabriele Cavallaro (g.cavallaro@fz-juelich.de) 
#         and Madita Willsch (m.willsch@fz-juelich.de)

import sys
import re
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cols
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,roc_curve,accuracy_score,auc

np.set_printoptions(precision=4, suppress=True)

def kernel(xn, xm, gamma=-1): # here (xn.shape: NxD, xm.shape: ...xD) -> Nx...
    if gamma == -1:
        return xn @ xm.T
    xn = np.atleast_2d(xn)
    xm = np.atleast_2d(xm)
    return np.exp(-gamma * np.sum((xn[:,None] - xm[None,:])**2, axis=-1)) # (N,1,D) - (1,...,D) -> (N,...,D) -> (N,...); see Hsu guide.pdf for formula

# B = base
# K = number of qubits per alpha
# E = shift of exponent
# decode binary -> alpha
def decode(binary, B=10, K=3, E=0):
    N = len(binary) // K
    Bvec = float(B) ** (np.arange(K)-E)
    return np.fromiter(binary,float).reshape(N,K) @ Bvec

# encode alpha -> binary with B and K (for each n, the binary coefficients an,k such that sum_k an,k B**k is closest to alphan)
def encode(alphas, B=10, K=3, E=0): # E allows for encodings with floating point numbers (limited precision of course)
    N = len(alphas)
    Bvec = float(B) ** (np.arange(K)-E) # B^(0-E) B^(1-E) B^(2-E) ... B^(K-1-E)
    allvals = np.array(list(map(lambda n : np.fromiter(bin(n)[2:].zfill(K),float,K), range(2**K)))) @ Bvec # [[0,0,0],[0,0,1],...] @ [1, 10, 100]
    return ''.join(list(map(lambda n : bin(n)[2:].zfill(K),np.argmin(np.abs(allvals[:,None] - alphas), axis=0))))

def encode_as_vec(alphas, B=10, K=3, E=0):
    return np.fromiter(encode(alphas,B,K,E), float)

def loaddataset(datakey):
    dataset = np.loadtxt(datakey, dtype=float, skiprows=1)
    return dataset[:,2:], dataset[:,1]  # data, labels

def save_json(filename, var):
    with open(filename,'w') as f:
        f.write(str(json.dumps(var, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)))

def eval_classifier(x, alphas, data, label, gamma, b=0): # evaluates the distance to the hyper plane according to 16.5.32 on p. 891 (Numerical Recipes); sign is the assigned class; x.shape = ...xD
    return np.sum((alphas * label)[:,None] * kernel(data, x, gamma), axis=0) + b

def eval_on_sv(x, alphas, data, label, gamma, C):
    return np.sum((alphas * (C-alphas) * label)[:,None] * kernel(data, x, gamma), axis=0)

def eval_offset_search(alphas, data, label, gamma, C, useavgforb=True): # search for the best offset
    maxacc=0
    b1=-9
    for i in np.linspace(-9,9,500):
        acc = accuracy_score(label,np.sign(eval_classifier(data, alphas, data, label, gamma, i)))
        if acc > maxacc:
            maxacc = acc
            b1=i
    maxacc=0
    b2=9
    reversed_space=np.linspace(-9,9,500)[::-1]
    for i in reversed_space:
        acc = accuracy_score(label,np.sign(eval_classifier(data, alphas, data, label, gamma, i)))
        if acc > maxacc:
            maxacc = acc
            b2=i
    return (b1+b2)/2

def eval_offset_MM(alphas, data, label, gamma, C, useavgforb=True): # evaluates offset b according to 16.5.37 (Mangasarian-Musicant variant) NOTE: does not seem to work with integer/very coarsely spaced alpha!
    return np.sum(alphas*label)

def eval_offset_avg(alphas, data, label, gamma, C, useavgforb=True): # evaluates offset b according to 16.5.33
    cross = eval_classifier(data, alphas, data, label, gamma) # cross[i] = sum_j aj yj K(xj, xi) (error in Numerical Recipes)
    if useavgforb:
        return np.sum(alphas * (C-alphas) * (label - cross)) / np.sum(alphas * (C-alphas))
        #return np.sum(label - cross) / num_sv
    else:  # this is actually not used, but we did a similar-in-spirit implementation in eval_finaltraining_avgscore.py
        if np.isclose(np.sum(alphas * (C-alphas)),0):
            print('no support vectors found, discarding this classifer')
            return np.nan
        bcandidates = [np.sum(alphas * (C-alphas) * (label - cross)) / np.sum(alphas * (C-alphas))]  # average according to NR should be the first candidate
        crosssorted = np.sort(cross)
        crosscandidates = -(crosssorted[1:] + crosssorted[:-1])/2  # each value between f(xi) and the next higher f(xj) is a candidate
        bcandidates += sorted(crosscandidates, key=lambda x:abs(x - bcandidates[0]))  # try candidates closest to the average first
        bnumcorrect = [(label == np.sign(cross + b)).sum() for b in bcandidates]
        return bcandidates[np.argmax(bnumcorrect)]

def eval_acc_auroc_auprc(label, score):  # score is the distance to the hyper plane (output from eval_classifier)
    precision,recall,_ = precision_recall_curve(label, score)
    return accuracy_score(label,np.sign(score)), roc_auc_score(label,score), auc(recall,precision)


################ This I/O functions are provided by http://hyperlabelme.uv.es/index.html ################ 

def dataread(filename):
    lasttag = 'description:'
    # Open file and locate lasttag
    f = open(filename, 'r')
    nl = 1
    for line in f:
        if line.startswith(lasttag): break
        nl += 1
    f.close()

    # Read data
    data = np.loadtxt(filename, delimiter=',', skiprows=nl)
    Y = data[:, 0]
    X = data[:, 1:]
    # Separate train/test
    Xtest = X[Y < 0, :]
    X = X[Y >= 0, :]
    Y = Y[Y >= 0, None]

    return X, Y, Xtest


def datawrite(path,method, dataset, Yp):
    filename = '{0}{1}_predictions.txt'.format(path, dataset)
    res = True
    try:
        with open(filename, mode='w') as f:
            f.write('{0} {1}'.format(method, dataset))
            for v in Yp:
                f.write(' {0}'.format(str(v)))
            f.write('\n')
    except Exception as e:
        print('Error', e)
        res = False
    return res

################ 


def write_samples(X, Y,path): 
    f = open(path,"w") 
    f.write("id label data \n") 
    for i in range(0,X.shape[0]):
        f.write(str(i)+" ")
        if(Y[i]==1):
            f.write("-1 ")
        else:
            f.write("1 ")
        for j in range(0,X.shape[1]):
            f.write(str(X[i,j])+" ")
        f.write("\n") 
    f.close() 