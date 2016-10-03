cimport c_python
cimport c_numpy
import numpy as np

from cython.parallel import prange

ctypedef unsigned int UINT32
ctypedef unsigned long int UINT64

cdef extern from "stdlib.h":
    void free(void* ptr)
    void* malloc(size_t size)
    void* realloc(void* ptr, size_t size)

cdef extern from "mtm_stats_core.h":
    int compute_counts(SparseBlockArray * sba_rows, int chunk_length, int i, int num_rows, SparseSetCounts * sparse_counts, int cutoff) nogil
    
    ctypedef struct SparseBlockArray:
        const UINT32* locs
        const UINT64* array
        UINT32 len
    
    ctypedef struct SparseSetCounts:
        UINT32 i
        UINT32 j
        UINT32 intersection_count
        UINT32 union_count

SPARSE_COUNTS_DTYPE = [('i', np.uint32),
                       ('j', np.uint32),
                       ('intersection_count', np.uint32),
                       ('union_count', np.uint32),]

cdef void set_SBA_from_py_dict(SparseBlockArray * sba, input_dict, sba_ind=0):
    '''Takes a python dictionary of arrays and fill a SparseBlockArray
       The expected input format for the input_dict is:
       {'locs': <numpy array of uint32>,
        'array': <numpy array of uint64>}
       '''
    cdef c_numpy.ndarray locs_cn = input_dict['locs']
    cdef c_numpy.ndarray array_cn = input_dict['array']
    
    sba[sba_ind].locs = <const UINT32 *> locs_cn.data
    sba[sba_ind].array = <const UINT64 *> array_cn.data
    sba[sba_ind].len = len(locs_cn)

def cy_mtm_stats(py_sba_array, chunk_length, cutoff=0):
    '''Run mtm_stats on 64-bit arrays
       Inputs:
        * py_sba_array: array of sparse block arrays (python format)
                        compressed version of the subset of B connected to each element of A
                        specifically, an array of dictionaries with fields 'array' and 'locs'
        * chunk_length: sba compression parameter
        * cutoff: maximum size of intersection to keep in the output
       
       Returns a numpy structured array with the following fields:
        * i, j: pair of indices into set A (set of interest)
        * intersection_count: number of elements in B that the 
                              A[i] and A[j] share in common
        * union_count: number of elements in B that the 
                       A[i] and A[j] connect to in total
    '''
    num_items = len(py_sba_array)
    
    # Map the numpy arrays directly to C pointers
    cdef SparseBlockArray * sba_pointer = <SparseBlockArray *>malloc(num_items * sizeof(SparseBlockArray))
    
    sparse_counts_list = []
    
    cdef int i
    
    for i in range(num_items):
        set_SBA_from_py_dict(sba_pointer, py_sba_array[i], sba_ind=i)
    
    # Run C compute_counts function on the generated pointers
    num_threads = 4
    
    # Set up results buffer and pointers:
    sparse_counts = np.zeros(num_items, dtype=SPARSE_COUNTS_DTYPE)
    cdef c_numpy.ndarray sparse_counts_cn = sparse_counts
    cdef SparseCounts * sparse_counts_pointer
    sparse_counts_pointer = <SparseCounts*> sparse_counts_cn.data
    
    cdef int chunk_length_c = chunk_length
    
    cdef int num_it = num_items
    for i in prange(num_it, nogil=True, chunksize=1, num_threads=num_threads, schedule='static'):
        compute_counts(sba_pointer[ia],
                       &sparse_counts_pointer,
                       chunk_length_c)

    # Return the results
    all_sparse_counts = np.concatenate(sparse_counts_list)
    return all_sparse_counts
