'''Sparse Block Array Compression'''
from __future__ import division

import numpy as np

def sba_compress(u8_array, chunk_size):
    '''Compress an array using the "Sparse Block Array" compression scheme
       The first argument must be a uint8 array
       The second argument is a chunk size in bytes
       Returns a dictionary:
         locs: int array of locations (size N)
         array: uint8 array of non-zero chunks (size N x chunk_size)
    '''
    num_bytes = len(u8_array)
    num_chunks = int(np.ceil(num_bytes / chunk_size))
    bulked = np.zeros((num_chunks, chunk_size), dtype=np.uint8)
    bulked.flat[:u8_array.shape[0]] = u8_array
    ind = np.where(np.sum(bulked, axis=1)!=0)
    return {'locs': np.array(ind[0], dtype=np.int32),
            'array': np.array(bulked[ind], dtype=np.uint8)}

def sba_compress_64(u64_array, chunk_size):
    '''Same as sba_compress except it takes a
       uint64 array
       and uses a 64 bit (8 byte) chunk size'''
    u8_array = u64_array.view(np.uint8)
    d = sba_compress(u8_array, chunk_size * 8)
    d['array'] = np.array(d['array'].view(np.uint64).flat)
    return d

def sba_compress_64_index_list(index_list, tmp_uint64_arr, chunk_length_64):
    '''Compress a set of indices into compressed SBA (Sparse Block Array) format

    index_list is a list of integer indices to be compressed
    tmp_uint64_arr needs to be a uint64 array with at least as many bits
    as the largest value in index_list
    this gets zeroed out and refilled by this function)
    '''
    tmp_uint64_arr *= 0
    for index in index_list:
        tmp_uint64_arr[index // 64] |= np.uint64(1 << (index % 64))
    return sba_compress_64(tmp_uint64_arr, chunk_length_64)

def sba_decompress(sba_dict, orig_length):
    '''This is SLOW, only useful for testing
       sba_dict has members 'locs' and 'array'
       as described in sba_compress'''
    num_locs, chunk_size = sba_dict['array'].shape
    num_bytes = orig_length
    num_chunks = np.ceil(num_bytes / chunk_size)
    bulked = np.zeros((num_chunks, chunk_size), dtype=np.uint8)
    bulked[sba_dict['locs']] = sba_dict['array']
    return np.array(bulked.flat[:orig_length])
