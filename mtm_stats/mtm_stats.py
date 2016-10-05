'''The main script'''
import numpy as np
from sparse_block_array import sba_compress_64
import cython_utils

def convert_connections_to_binary(connections):
    '''connections is a many-to-many mapping from set A to set B
       Returns a binary matrix where each item in set B gets mapped to a single bit and each item in set A gets a row of these bits'''
    setA = sorted({i[0] for i in connections})
    setB = sorted({i[1] for i in connections})
    mappingA = {p: i for i, p in enumerate(setA)}
    mappingB = {p: i for i, p in enumerate(setB)}
    
    lenB64 = int(np.ceil(len(setB) * 1. / 64))
    output = np.zeros((len(setA), lenB64), np.uint64)
    for a, b in connections:
        ia = mappingA[a]
        ib = mappingB[b]
        output[ia, ib // 64] |= np.uint64(1 << (ib % 64))
    return output

def mtm_stats(connections, chunk_length_64=1, cutoff=0):
    sba_list = [sba_compress_64(i, chunk_length_64)
                for i in convert_connections_to_binary(connections)]
    results = cython_utils.run_mtm_stats(sba_list, chunk_length_64, cutoff)
    return results
    

def get_Jaccard_index(connections):
    core.get_Jaccard_index

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
