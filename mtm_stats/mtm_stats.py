'''The main script'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import range
from future.utils import viewitems

# To update with any Cython changes, just run:
# python setup.py build_ext --inplace

import numpy as np
from .sparse_block_array import sba_compress_64, sba_compress_64_index_list
from . import cy_mtm_stats

def extract_sets_from_connections(connections):
    '''Get two sorted array sets from the connections tuples,
       one for the first elements and one for the second''' 
    setA = np.array(sorted({i[0] for i in connections}))
    setB = np.array(sorted({i[1] for i in connections}))
    return setA, setB

def convert_connections_to_binary(connections, setA, setB):
    '''connections is a many-to-many mapping from set A to set B
       Returns a binary matrix where each item in set B gets mapped to a single bit and each item in set A gets a row of these bits'''
    mappingA = {p: i for i, p in enumerate(setA)}
    mappingB = {p: i for i, p in enumerate(setB)}
    
    lenB64 = int(np.ceil(len(setB) / 64))
    output = np.zeros((len(setA), lenB64), np.uint64)
    for a, b in connections:
        ia = mappingA[a]
        ib = mappingB[b]
        output[ia, ib // 64] |= np.uint64(1 << (ib % 64))
    return output

def get_grouped_indices(connections, mappingA, mappingB):
    grouped = {}
    for a, b in connections:
        grouped.setdefault(mappingA[a],[]).append(mappingB[b])
    return grouped

def convert_connections_to_sba_list_space_efficient(connections, setA, setB, chunk_length_64):
    '''connections is a many-to-many mapping from set A to set B
       Returns a list of SBA compressed binary arrays where each item in set B gets mapped to a single bit and each item in set A gets a row of these bits'''
    mappingA = {p: i for i, p in enumerate(setA)}
    mappingB = {p: i for i, p in enumerate(setB)}
    
    lenB64 = int(np.ceil(len(setB) / 64))
    tmp_arr = np.empty(lenB64, np.uint64)
    grouped = get_grouped_indices(connections, mappingA, mappingB)
    sba_list = [None] * len(setA)
    for ia, ib_list in viewitems(grouped):
        sba_list[ia] = sba_compress_64_index_list(ib_list, tmp_arr, chunk_length_64)
    
    return sba_list

def _mtm_common(connections, chunk_length_64=1, dense_input=False):
    '''Common setup for static and partitioned-generator variants of
       mtm_stats
       There are three steps:
         * extracts the two sets
         * converts the connections to binary
         * compute the base counts
       
       Returns setA, setB, base_counts, and rows
       "rows" will be either a 2d rows_arr (when dense_input=True)
       or a sba_list (dense_input=False, DEFAULT)
       
       This is the data needed to perform the more expensive
       intersection counts calculation and the post-process union counts'''
    
    setA, setB = extract_sets_from_connections(connections)
    
    if dense_input:
        rows_arr = convert_connections_to_binary(connections, setA, setB)
        base_counts = cy_mtm_stats.cy_compute_counts_dense_input(rows_arr)
        rows = rows_arr
    else:
        sba_list = convert_connections_to_sba_list_space_efficient(connections, setA, setB, chunk_length_64)
        base_counts = cy_mtm_stats.cy_compute_counts(sba_list, chunk_length_64)
        rows = sba_list
    
    return setA, setB, base_counts, rows

def _mtm_intersection_counts(rows, chunk_length_64=1, indices_a=None, cutoff=0, start_j=0, upper_only=True, dense_input=False):
    '''The function that actually calls into cython for the intersection_counts
       Return the intersection_counts_list'''
    
    if dense_input:
        rows_arr = rows
        intersection_counts_list = cy_mtm_stats.cy_compute_intersection_counts_dense_input(rows_arr, indices_a, cutoff, start_j, upper_only)
    else:
        sba_list = rows
        intersection_counts_list = cy_mtm_stats.cy_compute_intersection_counts(sba_list, chunk_length_64, indices_a, cutoff, start_j, upper_only)
    
    return intersection_counts_list

def mtm_stats_raw(connections, chunk_length_64=1, indices_a=None, cutoff=0, start_j=0, upper_only=True, dense_input=False):
    '''The function that actually calls into cython
       Produces the sets from the connections,
       converts the connection to binary and compresses them into sba's
       and then performs the actual counts
       Returns:
           setA, setB, base_counts, intersection_counts_list'''
    
    setA, setB, base_counts, rows = _mtm_common(connections, chunk_length_64, dense_input)
    intersection_counts_list = _mtm_intersection_counts(rows, chunk_length_64, indices_a, cutoff, start_j, upper_only, dense_input)
    return setA, setB, base_counts, intersection_counts_list

def _partition_range(x, n):
    '''Return a generator for a series of chunked ranges that end at x
       partition_range(x, 1) <==> range(x)
       Examples:
          list(partition_range(19, 10)) -> [range(0, 10), range(10, 19)]
          list(partition_range(20, 10)) -> [range(0, 10), range(10, 20)]
          list(partition_range(21, 10)) -> [range(0, 10), range(10, 20), range(20, 21)]
       '''
    return (range(i, min(x, i+n)) for i in range(0, x, n))

def mtm_stats_raw_iterator(connections, partition_size, chunk_length_64=1, cutoff=0, start_j=0, upper_only=True, dense_input=False):
    '''This version of mtm_stats returns a generator instead of doing the
       actual intersection_counts calculation
       Each 
       Returns:
           setA, setB, base_counts, intersection_counts_generator'''
    
    setA, setB, base_counts, rows = _mtm_common(connections, chunk_length_64, dense_input)
    intersection_counts_generator = (_mtm_intersection_counts(rows, chunk_length_64, indices_a, cutoff, start_j, upper_only, dense_input)
                                     for indices_a in _partition_range(len(rows), partition_size))
    return setA, setB, base_counts, intersection_counts_generator

def get_base_counts_dict(base_counts, setA):
    return {setA[i]: p
            for i, p in enumerate(base_counts)}

def get_iu_counts_dict(base_counts, intersection_counts_list, setA):
    return {(setA[i], setA[j]): (ic, base_counts[i] + base_counts[j] - ic)
            for intersection_counts in intersection_counts_list
            for i, j, ic in intersection_counts}

def get_base_counts_gen(base_counts, setA):
    return ((setA[i], p)                         # (key, value)
            for i, p in enumerate(base_counts))

def get_iu_counts_gen(base_counts, intersection_counts_list, setA):
    return ((setA[i], setA[j], ic, base_counts[i] + base_counts[j] - ic)
            for intersection_counts in intersection_counts_list
            for i, j, ic in intersection_counts)

def mtm_stats(connections, chunk_length_64=1, indices_a=None, cutoff=0, start_j=0, upper_only=True, dense_input=False):
    '''Get base counts and intersection counts'''
    setA, setB, base_counts, intersection_counts_list = mtm_stats_raw(connections, chunk_length_64, indices_a, cutoff, start_j, upper_only, dense_input)
    base_counts_dict = get_base_counts_dict(base_counts, setA)
    iu_counts_dict = get_iu_counts_dict(base_counts, intersection_counts_list, setA)
    return base_counts_dict, iu_counts_dict

def mtm_stats_iterator(connections, partition_size, chunk_length_64=1, cutoff=0, start_j=0, upper_only=True, dense_input=False):
    '''Like mtm_stats, but returns generators instead of dicts for performance
       Returns:
         base_counts_generator:
            a generator that yeilds:
                (item_i, base_count_i)
         iu_counts_double_generator:
            a generator that yeilds a generator that yields:
                (item_i, item_j, intersection_count, union_count)
       
       Example for getting data out:
       for item_i, base_count_i in base_counts_generator:
           print("{} has {} things from set B".format(item_i, base_count_i))
        
       for iu_counts_gen in iu_counts_double_generator:
           for item_i, item_j, ic, uc in iu_counts_gen:
               print("{} and {} have {} things in common and {} things in total from set B".format(item_i, item_j, ic, uc))
       '''
    
    setA, setB, base_counts, intersection_counts_iterator = mtm_stats_raw_iterator(connections, partition_size, chunk_length_64, cutoff, start_j, upper_only, dense_input)
    base_counts_generator = get_base_counts_gen(base_counts, setA)
    
    iu_counts_double_generator = (get_iu_counts_gen(base_counts, intersection_counts_list, setA)
                                  for intersection_counts_list in intersection_counts_iterator)
    
    return base_counts_generator, iu_counts_double_generator

def get_Jaccard_index_from_sparse_connections(iu_counts_dict):
    return {k: ic / uc
            for k, (ic, uc) in viewitems(iu_counts_dict)}

def get_Jaccard_index(connections, chunk_length_64=1, indices_a=None, cutoff=0, start_j=0, upper_only=True, dense_input=False):
    base_counts_dict, iu_counts_dict = mtm_stats(connections, chunk_length_64, indices_a, cutoff, start_j, upper_only, dense_input)
    jaccard_index = get_Jaccard_index_from_sparse_connections(iu_counts_dict)
    return base_counts_dict, jaccard_index

def mtm_stats_from_iterator(connections, partition_size, chunk_length_64=1, cutoff=0, start_j=0, upper_only=True, dense_input=False):
    '''Same results as regular mtm_stats, but uses mtm_stats_iterator instead
       Mostly useful for testing, although it is not actually that different
       (faster or slower) than the original, so should probably just refactor to always do things this way'''
    base_counts_generator, iu_counts_double_generator = mtm_stats_iterator(connections, partition_size, chunk_length_64, cutoff, start_j, upper_only, dense_input)
    base_counts_dict = dict(base_counts_generator)
    iu_counts_dict = {(i, j): (ic, uc)
                      for iu_counts_generator in iu_counts_double_generator
                      for i, j, ic, uc in iu_counts_generator}
    return base_counts_dict, iu_counts_dict

if __name__ == '__main__':
    r = mtm_stats([('a1', 'b1'),
                   ('a1', 'b2'),
                   ('a1', 'b3'),
                   ('a2', 'b1'),
                   ('a2', 'b2'),
                   ('a3', 'b3'),
                   ('a4', 'b9'),])
    print(r[0])
    print(r[1])
