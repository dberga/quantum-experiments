# Created by Dennis Willsch (d.willsch@fz-juelich.de) 
# Modified by Gabriele Cavallaro (g.cavallaro@fz-juelich.de)
#         and Madita Willsch (m.willsch@fz-juelich.de)

import os
import sys
import re
import gc
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
import matplotlib.colors as cols
from utils import *

import shutil
import pickle
import numpy.lib.recfunctions as rfn
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system.composites import LazyFixedEmbeddingComposite
from dimod import BinaryQuadraticModel

def gen_svm_qubos(data,label,B,K,xi,gamma,E,path):

    N = len(data)
    
    if not os.path.isfile(path+'Q.npy'):
      Q = np.zeros((K*N,K*N))
    
      print(f'Creating the QUBO of size {Q.shape}')
      for n in range(N): # not optimized: size will not be so large and this way its more easily verifyable
          for m in range(N):
              for k in range(K):
                  for j in range(K):
                      Q[K*n+k,K*m+j] = .5 * B**(k+j-2*E) * label[n] * label[m] * (kernel(data[n], data[m], gamma) + xi)
                      if n == m and k == j:
                          Q[K*n+k,K*m+j] += - B**(k-E)


      Q = np.triu(Q) + np.tril(Q,-1).T # turn the symmetric matrix into upper triangular
    else:
      Q = np.load(path+'Q.npy')
    print(f'Extracting nodes and couplers')
    qubo_nodes = np.asarray([[n, n, Q[n,n]] for n in range(len(Q))])  # if not np.isclose(Q[n,n],0)]) NOTE: removed due to variable order!
    qubo_couplers = np.asarray([[n, m, Q[n,m]] for n in range(len(Q)) for m in range(n+1,len(Q)) if not np.isclose(Q[n,m],0)])
    qubo_couplers = qubo_couplers[np.argsort(-np.abs(qubo_couplers[:,2]))]


    print(f'Saving {len(qubo_nodes)} nodes and {len(qubo_couplers)} couplers for {path}')
    os.makedirs(path, exist_ok=True)
    np.save(path+'Q.npy', Q)
    np.savetxt(path+'qubo_nodes.dat', qubo_nodes, fmt='%g', delimiter='\t')
    np.savetxt(path+'qubo_couplers.dat', qubo_couplers, fmt='%g', delimiter='\t')
    
    return


def dwave_run_embedding(data,label,path_in,annealing_times,chain_strengths,em_id,solver='Advantage_system1.1'): #em_id is a label for the embedding. In this way, it is possible to re-run a previously computed and stored embedding with e.g. different chain strength and/or annealing time
    
    MAXRESULTS = 20 # NOTE: to save space only 20 best results
    match = re.search('run_([^/]*)_B=(.*)_K=(.*)_xi=(.*)_E=(.*)_gamma=([^/]*)', path_in) 

    data_key = match.group(1)
    B = int(match.group(2))
    K = int(match.group(3))
    xi = float(match.group(4))
    gamma = float(match.group(6))
    E = int(match.group(5))

    path = path_in+ ('/' if path_in[-1] != '/' else '')
    qubo_couplers = np.loadtxt(path+'qubo_couplers.dat')
    qubo_nodes = np.loadtxt(path+'qubo_nodes.dat')
    qubo_nodes = np.array([[i,i,(qubo_nodes[qubo_nodes[:,0]==i,2][0] if i in qubo_nodes[:,0] else 0.)] for i in np.arange(np.concatenate((qubo_nodes,qubo_couplers))[:,[0,1]].max()+1)])  # to make sure every (i,i) occurs in the qubo in increasing order such that the variable order in BinaryQuadraticModel is consistent (see locate wrongenergies-* github issue)
    maxcouplers = len(qubo_couplers) ## POSSIBLE INPUT if len(sys.argv) <= 2 else int(sys.argv[2])

    if not 'train' in data_key:
        raise Exception(f'careful: datakey={data_key} => youre trying to train on a validation / test set!')

    couplerslist = [min(7500,maxcouplers)] # The 7500 here is more or less arbitrary and may need to be adjusted. Just to be sure that the number of couplers is not larger than the number of physical couplers (EmbeddingComposite does not seem to check for this)
    for trycouplers in [5000, 2500, 2000, 1800, 1600, 1400, 1200, 1000, 500]:
        if maxcouplers > trycouplers:
            couplerslist += [trycouplers]


    sampler = LazyFixedEmbeddingComposite(DWaveSampler(solver=solver)) # use the same embedding for all chain strengths and annealing times
    for n in range(0,len(chain_strengths)):
        for m in range(0,len(annealing_times)):
            if n==0 and m==0:
                for couplers in couplerslist:  # try to reduce as little couplers as necessary to find an embedding
                    Q = { (q[0], q[1]): q[2] for q in np.vstack((qubo_nodes, qubo_couplers[:couplers])) }
                    maxstrength=np.max( np.abs( list( Q.values() ) ) )
                    pathsub = path + f'result_couplers={couplers}/'
                    os.makedirs(pathsub, exist_ok=True)
                    embedding_file_name=pathsub+f'embedding_id{em_id}'
                    if os.path.isfile(embedding_file_name):
                        embedding_file=open(embedding_file_name,'r')
                        embedding_data=eval(embedding_file.read())
                        embedding_file.close()
                        sampler._fix_embedding(embedding_data)
                    print(f'running {pathsub} with {len(qubo_nodes)} nodes and {couplers} couplers for embedding {em_id}')
    
                    ordering = np.array(list(BinaryQuadraticModel.from_qubo(Q)))
                    if not (ordering == np.arange(len(ordering),dtype=ordering.dtype)).all():
                        print(f'WARNING: variables are not correctly ordered! path={path} ordering={ordering}')

                    try:
                        print(f'Running chain strength {chain_strengths[n]} and annealing time {annealing_times[m]}\n')
                        response = sampler.sample_qubo(Q, num_reads=5000, annealing_time=annealing_times[0], chain_strength=chain_strengths[0]*maxstrength)
                        if not os.path.isfile(embedding_file_name):
                            embedding_file=open(embedding_file_name,'w')
                            print('Embedding found. Saving...')
                            embedding_file.write(repr(sampler.embedding))
                            embedding_file.close()
            
                    except ValueError as v:
                        print(f' -- no embedding found, removing {pathsub} and trying less couplers')
                        shutil.rmtree(pathsub)
                        sampler = LazyFixedEmbeddingComposite(DWaveSampler(solver=solver))
                        continue
                    break
            else:
                print(f'Running chain strength {chain_strengths[n]} and annealing time {annealing_times[m]}')
                response = sampler.sample_qubo(Q, num_reads=5000, annealing_time=annealing_times[m], chain_strength=chain_strengths[n]*maxstrength)

            pathsub_ext=pathsub+f'embedding{em_id}_rcs{chain_strengths[n]}_ta{annealing_times[m]}_'
            save_json(pathsub_ext+'info.json', response.info) # contains response.info
            #NOTE left out: pickle.dump(response, open(pathsub+'response.pkl','wb')) # contains full response incl. response.record etc; can be loaded with pickle.load(open('response.pkl','rb'))

            samples = np.array([''.join(map(str,sample)) for sample in response.record['sample']]) # NOTE: it would be safer to use the labeling from record.data() for the qubit variable order
            unique_samples, unique_idx, unique_counts = np.unique(samples, return_index=True, return_counts=True) # unfortunately, num_occurrences seems not to be added up after unembedding
            unique_records = response.record[unique_idx]
            result = rfn.merge_arrays((unique_samples, unique_records['energy'], unique_counts, unique_records['chain_break_fraction']))  # see comment on chain_strength above
            result = result[np.argsort(result['f1'])]
            np.savetxt(pathsub_ext+'result.dat', result[:MAXRESULTS], fmt='%s', delimiter='\t', header='\t'.join(response.record.dtype.names), comments='') # load with np.genfromtxt(..., dtype=['<U2000',float,int,float], names=True, encoding=None)

            alphas = np.array([decode(sample,B,K,E) for sample in result['f0'][:MAXRESULTS]])
            np.save(pathsub_ext+f'alphas.npy', alphas)
            gc.collect()
    
    return pathsub


def eval_run_trainaccuracy(path_in):

    regex = 'run([^/]*)_B=(.*)_K=(.*)_xi=(.*)_E=(.*)_gamma=([^/]*)/result_couplers.*/?$'
    match = re.search(regex, path_in)
 
    path = path_in + ('/' if path_in[-1] != '/' else '')
    data_key = match.group(1)
    B = int(match.group(2))
    K = int(match.group(3))
    xi = float(match.group(4))
    gamma = float(match.group(6))
    E = int(match.group(5))
    data,label = loaddataset(data_key)

    alphas_file = path+f'alphas{data_key}_gamma={gamma}.npy'
    if not os.path.isfile(alphas_file):
        print('result '+alphas_file+' doesnt exist, exiting')
        sys.exit(-1)

    alphas = np.atleast_2d(np.load(alphas_file))
    nalphas = len(alphas)
    assert len(data) == alphas.shape[1], "alphas do not seem to be for the right data set?)"

    result = np.genfromtxt(path+'result.dat', dtype=['<U2000',float,int,float], names=True, encoding=None, max_rows=nalphas)

    Cs = [100, 10, (B**np.arange(K-E)).sum(), 1.5]
    evaluation = np.zeros(nalphas, dtype=[('sum_antn',float)]+[(f'acc(C={C})',float) for C in Cs])

    for n,alphas_n in enumerate(alphas):
        evaluation[n]['sum_antn'] = (label * alphas_n).sum()
        for j,field in enumerate(evaluation.dtype.names[1:]):
            b = eval_offset_avg(alphas_n, data, label, gamma, Cs[j]) # NOTE: this is NAN if no support vectors were found, see TODO file
            label_predicted = np.sign(eval_classifier(data, alphas_n, data, label, gamma, b)) # NOTE: this is only train accuracy! (see eval_result_roc*)
            evaluation[n][field] = (label == label_predicted).sum() / len(label)

    result_evaluated = rfn.merge_arrays((result,evaluation), flatten=True)
    fmt = '%s\t%.3f\t%d\t%.3f' + '\t%.3f'*len(evaluation.dtype.names)
    #NOTE: left out
    # np.savetxt(path+'result_evaluated.dat', result_evaluated, fmt=fmt, delimiter='\t', header='\t'.join(result_evaluated.dtype.names), comments='') # load with np.genfromtxt(..., dtype=['<U2000',float,int,float,float,float,float,float], names=True, encoding=None)

    print(result_evaluated.dtype.names)
    print(result_evaluated)
    
    

def eval_run_rocpr_curves(path_data_key,path_in,plotoption):  

    regex = 'run([^/]*)_B=(.*)_K=(.*)_xi=(.*)_E=(.*)_gamma=([^/]*)/result_couplers.*/?$' 
    match = re.search(regex, path_in) 

    path = path_in + ('/' if path_in[-1] != '/' else '')
    data_key = match.group(1)
    B = int(match.group(2))
    K = int(match.group(3))
    xi = float(match.group(4))
    gamma = float(match.group(6))
    E = int(match.group(5))
    data,label = loaddataset(path_data_key+data_key)
  
    dwavesolutionidx=0
    C=(B**np.arange(K-E)).sum()

    if 'calibtrain' in data_key:
        testname = 'Validation'
        datatest,labeltest = loaddataset(path_data_key+data_key.replace('calibtrain','calibval'))
    else:
        print('be careful: this does not use the aggregated bagging classifier but only the simple one as in calibration')
        testname = 'Test'
        datatest,labeltest = loaddataset(re.sub('train(?:set)?[0-9]*(?:bag)[0-9]*','test',data_key))

    alphas_file = path+f'alphas{data_key}_gamma={gamma}.npy'
    if not os.path.isfile(alphas_file):
        print('result '+alphas_file+' doesnt exist, exiting')
        sys.exit(-1)

    alphas = np.atleast_2d(np.load(alphas_file))
    nalphas = len(alphas)
    assert len(data) == alphas.shape[1], "alphas do not seem to be for the right data set?)"

    print('idx   \tsum_antn\ttrainacc\ttrainauroc\ttrainauprc\ttestacc  \ttestauroc\ttestauprc')
    
    trainacc_all=np.zeros([nalphas])
    trainauroc_all=np.zeros([nalphas])
    trainauprc_all=np.zeros([nalphas])
    
    testacc_all=np.zeros([nalphas])
    testauroc_all=np.zeros([nalphas])
    testauprc_all=np.zeros([nalphas])

    for i in range(nalphas):
        alphas_n = alphas[i]
        b = eval_offset_avg(alphas_n, data, label, gamma, C) # NOTE: this is NAN if no support vectors were found, see TODO file
        score = eval_classifier(data, alphas_n, data, label, gamma, b)
        scoretest = eval_classifier(datatest, alphas_n, data, label, gamma, b)
        trainacc,trainauroc,trainauprc = eval_acc_auroc_auprc(label,score)
        testacc,testauroc,testauprc = eval_acc_auroc_auprc(labeltest,scoretest)
        
        trainacc_all[i]=trainacc
        trainauroc_all[i]=trainauroc
        trainauprc_all[i]=trainauprc
        testacc_all[i]=testacc
        testauroc_all[i]=testauroc
        testauprc_all[i]=testauprc

        print(f'{i}\t{(label*alphas_n).sum():8.4f}\t{trainacc:8.4f}\t{trainauroc:8.4f}\t{trainauprc:8.4f}\t{testacc:8.4f}\t{testauroc:8.4f}\t{testauprc:8.4f}')
                   

    # plot code starts here
    if plotoption != 'noplotsave':
        alphas_n = alphas[dwavesolutionidx] # plot only the requested
        b = eval_offset_avg(alphas_n, data, label, gamma, C) # NOTE: this is NAN if no support vectors were found, see TODO file
        score = eval_classifier(data, alphas_n, data, label, gamma, b)
        scoretest = eval_classifier(datatest, alphas_n, data, label, gamma, b)

        # roc curve
        plt.figure(figsize=(6.4,3.2))
        plt.subplot(1,2,1)
        plt.subplots_adjust(top=.95, right=.95, bottom=.15, wspace=.3)
        fpr, tpr, thresholds = roc_curve(labeltest, scoretest)
        auroc = roc_auc_score(labeltest, scoretest)
        plt.plot(fpr, tpr, label='AUROC=%0.3f' % auroc, color='g')
        plt.fill_between(fpr, tpr, alpha=0.2, color='g', step='post')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.title('Receiver Operating Curve')
        plt.legend(loc="lower right")
        # pr curve
        plt.subplot(1,2,2)
        precision, recall, _ = precision_recall_curve(labeltest, scoretest)
        auprc = auc(recall, precision)
        plt.step(recall, precision, color='g', where='post',
            label='AUPRC=%0.3f' % auprc)
        plt.fill_between(recall, precision, alpha=0.2, color='g', step='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])
        #plt.title('PR curve')
        plt.legend(loc="lower right")

        # save the data for gnuplot
        savename = f'{path.replace("/","_")}{dwavesolutionidx}'
        #with open('results/rocpr_curves/'+savename,'w') as out:
        with open(path_in+savename,'w') as out:
            out.write(f'AUROC\t{auroc:0.3f}\t# ROC:FPR,TPR\n')
            assert len(fpr) == len(tpr)
            for i in range(len(fpr)):
                out.write(f'{fpr[i]}\t{tpr[i]}\n')
            out.write(f'\n\nAUPRC\t{auprc:0.3f}\t# PRC:Recall,Precision\n')
            assert len(recall) == len(precision)
            for i in range(len(recall)):
                out.write(f'{recall[i]}\t{precision[i]}\n')
            print(f'saved data for {savename}')

        if plotoption == 'saveplot':
            savefigname = path_in+savename+'.pdf'
            plt.savefig(savefigname)
            print(f'saved as {savefigname}')
        else:
            plt.show()
            
    
    return np.average(trainacc_all), np.average(trainauroc_all), np.average(trainauprc_all) ,np.average(testacc_all), np.average(testauroc_all), np.average(testauprc_all)


# previously we used AUROC and AUPRC as metrics, now replaced by F1 score -> this function here is maybe not necessary anymore?
# returns the average of the results of all single classifiers and the result of the averaged classifer (for train and validation data)
# if validation = False, the "validation" results are all 0 (except energy which is only computed and returned for train data anyway)
# showplot option only works for toy model data, thus False by default
def eval_run_rocpr_curves_embedding_train(path_data_key,path_in,cs,ta,idx,max_alphas=0,validation=False,showplot=False):

    regex = '.*run([^/]*)_B=(.*)_K=(.*)_xi=(.*)_E=(.*)_gamma=([^/]*)/result_couplers.*/?$' 
    match = re.search(regex, path_in) 

    path = path_in + ('/' if path_in[-1] != '/' else '')
    data_key = match.group(1)
    B = int(match.group(2))
    K = int(match.group(3))
    xi = float(match.group(4))
    gamma = float(match.group(6))
    E = int(match.group(5))
    data,label = loaddataset(path_data_key+data_key)
    if validation:
        data_val,label_val = loaddataset(path_data_key+data_key.replace('calibtrain','calibval'))
  
    dwavesolutionidx=0
    C=(B**np.arange(K-E)).sum()

    alphas_file = path+f'embedding{idx}_rcs{cs}_ta{ta}_alphas{data_key}_gamma={gamma}.npy'
    energy_file = path+f'embedding{idx}_rcs{cs}_ta{ta}_result.dat'
    energy_file_dat=np.genfromtxt(energy_file);
    energies=np.transpose(energy_file_dat[1:])[1]
    if not os.path.isfile(alphas_file):
        print('result '+alphas_file+' doesnt exist, exiting')
        sys.exit(-1)

    alphas = list(np.atleast_2d(np.load(alphas_file)))
    for i in reversed(range(len(alphas))):
        if all(np.logical_or(np.isclose(alphas[i],0),np.isclose(alphas[i],C))) or np.isclose(np.sum(alphas[i] * (C-alphas[i])),0): #remove cases where no support vectors are found (all alpha=0 or only slack variables all alpha=0 or alpha=C) -> numerical recipes Eq. 16.5.24
            print(f'Deleting alphas[{i}].')
            del alphas[i]
    alphas = np.array(alphas)
    if max_alphas == 0 or max_alphas > len(alphas):
        nalphas = len(alphas)
    else:
        nalphas = max_alphas
    assert len(data) == alphas.shape[1], "alphas do not seem to be for the right data set?)"
    
    trainacc_all=np.zeros([nalphas])
    trainauroc_all=np.zeros([nalphas])
    trainauprc_all=np.zeros([nalphas])
    testacc_all=np.zeros([nalphas])
    testauroc_all=np.zeros([nalphas])
    testauprc_all=np.zeros([nalphas])
    alphas_avg = np.zeros(len(alphas[0]))
    energies2 = np.zeros([nalphas])

    for i in range(nalphas):
        alphas_n = alphas[i]
        alphas_avg += alphas_n
        b = eval_offset_avg(alphas_n, data, label, gamma, C) # NOTE: this is NAN if no support vectors were found, see TODO file
        score = eval_classifier(data, alphas_n, data, label, gamma, b)
        trainacc,trainauroc,trainauprc = eval_acc_auroc_auprc(label,score)
        
        trainacc_all[i] = trainacc
        trainauroc_all[i] = trainauroc
        trainauprc_all[i] = trainauprc
        energies2[i] = compute_energy(alphas_n, data, label, gamma, xi)
        
        if validation:
            scoretest = eval_classifier(data_val, alphas_n, data, label, gamma, b)
            testacc,testauroc,testauprc = eval_acc_auroc_auprc(label_val,scoretest)
        
            testacc_all[i] = testacc
            testauroc_all[i] = testauroc
            testauprc_all[i] = testauprc

    alphas_avg = alphas_avg/nalphas
    b = eval_offset_avg(alphas_avg, data, label, gamma, C) # NOTE: this is NAN if no support vectors were found, see TODO file
    score = eval_classifier(data, alphas_avg, data, label, gamma, b)
    trainacc,trainauroc,trainauprc = eval_acc_auroc_auprc(label,score)
    testacc = 0
    testauroc = 0
    testauprc = 0
    if validation:
        scoretest = eval_classifier(data_val, alphas_avg, data, label, gamma, b)
        testacc,testauroc,testauprc = eval_acc_auroc_auprc(label_val,scoretest)
    energy = compute_energy(alphas_avg, data, label, gamma,xi)
    print(alphas_avg)
    
    if showplot:
        if validation:
            plot_result_val(alphas_avg, data, label, gamma, C, [-4,4], title=f'ta = {ta}, rcs = {cs:0.1f}, emb = {idx}, energy {energy:0.4f},\nacc = {trainacc:0.4f}, auroc = {trainauroc:0.4f},  auprc = {trainauprc:0.4f},\ntestacc = {testacc:0.4f}, testauroc = {testauroc:0.4f},  testauprc = {testauprc:0.4f}', filled=True, validation=True, data_val=data_val, label_val=label_val)
        else:
            plot_result_val(alphas_avg, data, label, gamma, C, [-4,4], title=f'ta = {ta}, rcs = {cs:0.1f}, emb = {idx}, energy {energy:0.4f},\nacc = {trainacc:0.4f}, auroc = {trainauroc:0.4f},  auprc = {trainauprc:0.4f}', filled=True)

    return np.average(trainacc_all), np.average(trainauroc_all), np.average(trainauprc_all), trainacc, trainauroc, trainauprc, np.average(energies2), energy, np.average(testacc_all), np.average(testauroc_all), np.average(testauprc_all), testacc, testauroc, testauprc

def compute_energy(alphas, data, label, gamma, xi):
    energy = 0
    sv = alphas*label
    for n in range(len(data)):
        for m in range(len(data)):
            energy += 0.5*sv[n]*sv[m]*kernel(data[n],data[m],gamma)
    energy -= np.sum(alphas)
    energy += 0.5*xi*(np.sum(sv))**2 # NOTE: the 1/2 here is only there because we have it in the svm paper this way
    return energy

# for plotting the toy model data with classifier and support vectors
def plot_result(alphas, data, label, gamma, C, xylim=[-2.0, 2.0], notitle=False, filled=False, title = "", save=""):
    b = eval_offset_avg(alphas, data, label, gamma, C)
    result = np.sign(eval_classifier(data, alphas, data, label, gamma, b))
    w = np.array([0.0,0.0])
    for l in range(0,len(label)):
        w += alphas[l] * label[l] * data[l]
    len_w = np.sqrt(w[0]**2+w[1]**2)
    w = w/len_w
    print(f'w = ( {w[0]} , {w[1]} )')
    w = -b/len_w*w
    print(f'b = {b}')

    xsample = np.arange(xylim[0], xylim[1]+.01, .05)
    x1grid, x2grid = np.meshgrid(xsample, xsample)
    X = np.vstack((x1grid.ravel(), x2grid.ravel())).T # ...xD for kernel contraction
    FX = eval_classifier(X, alphas, data, label, gamma, b).reshape(len(xsample), len(xsample))

    #plt.pcolor(x1grid, x2grid, FX, cmap='coolwarm')
    plt.contourf(x1grid, x2grid, FX, [-5.,-4.5,-4.,-3.5,-3.,-2.5,-2.,-1.5,-1.,-0.5,0.,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.], cmap='seismic', extend='both',alpha=0.8)
    plt.contour(x1grid, x2grid, FX, [0.], linewidths=3, colors='black')
    plt.arrow(0,0,w[0],w[1], linewidth=2, length_includes_head=True)
    ax=plt.gca()
    ax.set_aspect('equal')
    if not title == "":
        plt.title(title)
    if not notitle:
        plt.title('acc = %g' % ((result == label).sum() / len(label)))

    if not filled:
        plt.scatter(data[result==1][:,0], data[result==1][:,1], c='r', marker=(8,2,0), linewidth=0.5)
        plt.scatter(data[result!=1][:,0], data[result!=1][:,1], c='b', marker='+', linewidth=1)
        plt.scatter(data[label==1][:,0], data[label==1][:,1], edgecolors='r', marker='s', linewidth=0.5, facecolors='none')
        plt.scatter(data[label!=1][:,0], data[label!=1][:,1], edgecolors='b', marker='o', linewidth=1, facecolors='none')
    else:
        plt.scatter(data[label==1][:,0], data[label==1][:,1], edgecolors=cols.CSS4_COLORS['darkorange'], marker='s', linewidth=1, facecolors=cols.CSS4_COLORS['red'])
        plt.scatter(data[label!=1][:,0], data[label!=1][:,1], edgecolors=cols.CSS4_COLORS['deepskyblue'], marker='D', linewidth=1, facecolors=cols.CSS4_COLORS['blue'])
        plt.scatter(data[alphas>0][:,0], data[alphas>0][:,1], edgecolors='k', marker='o', linewidth=2, facecolors='none')
        plt.scatter(data[alphas==C][:,0], data[alphas==C][:,1], edgecolors='w', marker='o', linewidth=2, facecolors='none')

    plt.xlim(*xylim)
    plt.ylim(*xylim)
    plt.show()
    
    if not save == "":
        plt.savefig(save+".svg")

def predict(datatest,data,label,alphas,path_in):  

    regex = 'run([^/]*)_B=(.*)_K=(.*)_xi=(.*)_E=(.*)_gamma=([^/]*)/result_couplers.*/?$'  
    match = re.search(regex, path_in) 

    path = path_in + ('/' if path_in[-1] != '/' else '')
    data_key = match.group(1)
    B = int(match.group(2))
    K = int(match.group(3))
    xi = float(match.group(4))
    gamma = float(match.group(6))
    E = int(match.group(5))
    
    C=(B**np.arange(K-E)).sum()
    
    # Compute the mean of the alphas 
    alphas_avg=np.mean(alphas,axis=0)
    
    b = eval_offset_avg(alphas_avg, data, label, gamma, C) # NOTE: this is NAN if no support vectors were found, see TODO file
    
    scoretest = eval_classifier(datatest, alphas_avg, data, label, gamma, b)
      
    return scoretest
