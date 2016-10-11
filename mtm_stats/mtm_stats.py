'''The main script'''

# To update with any Cython changes, just run:
# python setup.py build_ext --inplace

import numpy as np
from sparse_block_array import sba_compress_64
import cy_mtm_stats

def extract_sets_from_connections(connections):
    '''Get two sorted sets from the connections tuples,
       one for the first elements and one for the second''' 
    setA = sorted({i[0] for i in connections})
    setB = sorted({i[1] for i in connections})
    return setA, setB

def convert_connections_to_binary(connections, setA, setB):
    '''connections is a many-to-many mapping from set A to set B
       Returns a binary matrix where each item in set B gets mapped to a single bit and each item in set A gets a row of these bits'''
    mappingA = {p: i for i, p in enumerate(setA)}
    mappingB = {p: i for i, p in enumerate(setB)}
    
    lenB64 = int(np.ceil(len(setB) * 1. / 64))
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
    
    lenB64 = int(np.ceil(len(setB) * 1. / 64))
    tmp_arr = np.empty(lenB64, np.uint64)
    grouped = get_grouped_indices(connections, mappingA, mappingB)
    sba_list = [None] * len(setA)
    for ia, ib_list in grouped.iteritems():
        tmp_arr *= 0
        for ib in ib_list:
            tmp_arr[ib // 64] |= np.uint64(1 << (ib % 64))
        sba_list[ia] = sba_compress_64(tmp_arr, chunk_length_64)
    
    return sba_list

def mtm_stats_raw(connections, chunk_length_64=1, cutoff=0):
    '''The function that actually calls into cython
       Produces the sets from the connections,
       converts the connection to binary and compresses them into sba's
       and then performs the actual counts
       Returns:
           setA, setB, base_counts, sparse_counts'''
    setA, setB = extract_sets_from_connections(connections)
    #sba_list = [sba_compress_64(i, chunk_length_64)
    #            for i in convert_connections_to_binary(connections, setA, setB)]
    sba_list = convert_connections_to_sba_list_space_efficient (connections, setA, setB, chunk_length_64)
    base_counts, sparse_counts = cy_mtm_stats.cy_mtm_stats(sba_list, chunk_length_64, cutoff)
    return setA, setB, base_counts, sparse_counts

def get_dicts_from_array_outputs(base_counts, sparse_counts, setA):
    base_counts_dict = {setA[i]: p
                        for i, p in enumerate(base_counts)}
    sparse_counts_dict = {(setA[i], setA[j]): (ic, uc)
                          for i, j, ic, uc in sparse_counts}
    return base_counts_dict, sparse_counts_dict

def mtm_stats(connections, chunk_length_64=1, cutoff=0):
    setA, setB, base_counts, sparse_counts = mtm_stats_raw(connections, chunk_length_64=1, cutoff=0)
    base_counts_dict, sparse_counts_dict = get_dicts_from_array_outputs(base_counts, sparse_counts, setA)
    return base_counts_dict, sparse_counts_dict

def get_Jaccard_index_from_sparse_connections(sparse_counts_dict):
    return {k: ic * 1. / uc
            for k, (ic, uc) in sparse_counts_dict.iteritems()}

def get_Jaccard_index(connections, chunk_length_64=1, cutoff=0):
    base_counts_dict, sparse_counts_dict = mtm_stats(connections, chunk_length_64=1, cutoff=0)
    jaccard_index = get_Jaccard_index_from_sparse_connections(sparse_counts_dict)
    return base_counts_dict, jaccard_index

if __name__ == '__main__':
    r = mtm_stats([('a1', 'b1'),
                   ('a1', 'b2'),
                   ('a1', 'b3'),
                   ('a2', 'b1'),
                   ('a2', 'b2'),
                   ('a3', 'b3'),
                   ('a4', 'b9'),])
    print r[0]
    print r[1]
