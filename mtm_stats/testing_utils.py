'''Utility functions for use in testing'''

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
    return connections

def run_timing_test(*args, **kwds):
    '''Generate a test set and run mtm_stats
       Time both steps and print the output
       kwds: verbose=True -> print the test set and the result from mtm_stats
       
       all other args and kwds are passed to generate_test_set
       '''
    verbose = kwds.pop('verbose', True)
    
    t = time.time()
    connections = generate_test_set(*args, **kwds)
    generate_time = time.time()-t
    if verbose:
        print connections
    
    t = time.time()
    bc_dict, sc_dict = mtm_stats.mtm_stats(connections)
    process_time = time.time()-t
    if verbose:
        print bc_dict
        print sc_dict
    
    print generate_time, process_time
    return generate_time, process_time
