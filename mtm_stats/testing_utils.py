'''Utility functions for use in testing'''
from __future__ import print_function
from __future__ import absolute_import
from builtins import zip
from builtins import range
from future.utils import viewitems

import time
import numpy as np
import mtm_stats

def generate_test_set(sizeA=10000,
                      sizeB = 10000000,
                      num_connections = 1000000,
                      beta_paramsA=(0.2, 1),
                      beta_paramsB=(0.2, 1),
                      seed=0,
                     ):
    '''Generate a test set to pass to mtm_stats
       Be aware that setA and setB may not match exactly to the setA
       and setB computed in mtm_stats (those will be subsets)'''
    np.random.seed(seed)
    setA_full = ['a{}'.format(i) for i in range(sizeA)]
    setB_full = ['b{}'.format(i) for i in range(sizeB)]
    divsum = lambda x: x * 1. / x.sum()
    weightsA = divsum(np.random.beta(beta_paramsA[0], beta_paramsA[1], size=sizeA))
    weightsB = divsum(np.random.beta(beta_paramsB[0], beta_paramsB[1], size=sizeB))
    a_inds = np.random.choice(sizeA, num_connections, p=weightsA)
    b_inds = np.random.choice(sizeB, num_connections, p=weightsB)
    a_list = [setA_full[i] for i in a_inds]
    b_list = [setB_full[i] for i in b_inds]
    connections = zip(a_list, b_list)
    return sorted(set(connections)) # Make sure everything is unique, etc

def run_timing_test(*args, **kwds):
    '''Generate a test set and run mtm_stats
       Time both steps and print the output
       kwds:
           verbose=True -> print the test set and the result from mtm_stats
           chunk_length_64=True -> passed to mtm_stats
           cutoff=0 -> passed to mtm_stats
       
       all other args and kwds are passed to generate_test_set
       '''
    verbose = kwds.pop('verbose', False)
    chunk_length_64 = kwds.pop('chunk_length_64', 1)
    cutoff = kwds.pop('cutoff', 0)
    
    t = time.time()
    connections = generate_test_set(*args, **kwds)
    generate_time = time.time()-t
    if verbose:
        print(connections)
    
    t = time.time()
    setA, setB, base_counts, intersection_counts = mtm_stats.mtm_stats_raw(connections)
    process_time = time.time()-t
    if verbose:
        print(setA, setB, base_counts, intersection_counts)
    
    # Printing/returning section:
    sizeA = kwds.pop('sizeA', 10000)
    sizeB = kwds.pop('sizeB', 10000000)
    num_connections = kwds.pop('num_connections', 1000000)
    print(sizeA, sizeB, num_connections, generate_time, process_time)
    return generate_time, process_time

def naive_counts(connections):
    '''connections is a many-to-many mapping from set A to set B
       Uses a very naive algorithm to compute the intersection and union counts
       Returns a dict of tuples keyed on tuples'''
    setA, setB = mtm_stats.extract_sets_from_connections(connections)
    mappingA = {p: i for i, p in enumerate(setA)}
    mappingB = {p: i for i, p in enumerate(setB)}
    
    grouped = mtm_stats.get_grouped_indices(connections, mappingA, mappingB)
    
    base_counts_dict = {setA[ia]: len(ib_list)
                        for ia, ib_list in viewitems(grouped)}
    
    sparse_counts_dict = {}
    for i in range(len(setA)):
        for j in range(i):
            ai, aj = setA[i], setA[j]
            sparse_counts_dict[(aj, ai)] = (len(set(grouped[i]) & set(grouped[j])),
                                            len(set(grouped[i]) | set(grouped[j])))
    sparse_counts_dict = {k: v for k, v in viewitems(sparse_counts_dict)
                               if v[0] > 0}
    
    return base_counts_dict, sparse_counts_dict

def naivest_counts(connections):
    '''connections is a many-to-many mapping from set A to set B
       Uses a very naive algorithm to compute the intersection and union counts
       Returns a dict of tuples keyed on tuples
       Even slower version that doesn't use indexes'''
    setA, setB = mtm_stats.extract_sets_from_connections(connections)
    
    grouped = {}
    for a, b in connections:
        grouped.setdefault(a,[]).append(b)
    
    base_counts_dict = {a: len(b_list)
                        for a, b_list in viewitems(grouped)}
    
    sparse_counts_dict = {}
    for i in range(len(setA)):
        for j in range(i):
            ai, aj = setA[i], setA[j]
            sparse_counts_dict[(aj, ai)] = (len(set(grouped[ai]) & set(grouped[aj])),
                                            len(set(grouped[ai]) | set(grouped[aj])))
    sparse_counts_dict = {k: v for k, v in viewitems(sparse_counts_dict)
                               if v[0] > 0}
    return base_counts_dict, sparse_counts_dict

def is_naive_same(connections, print_time=False, verbose=False):
    '''Test mtm_stats against naive_counts'''
    t = time.time()
    mtm_result = mtm_stats.mtm_stats(connections)
    if print_time:
        print('mtm_stats speed:', time.time()-t)
    t = time.time()
    naive_result = naive_counts(connections)
    if print_time:
        print('naive speed:', time.time()-t)
    
    if verbose:
        print('connections', connections)
        print('mtm_stats result', naive_result)
        print('naive result', naive_result)
    return naive_result == mtm_result
