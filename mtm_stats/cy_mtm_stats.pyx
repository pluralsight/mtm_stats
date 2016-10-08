cimport c_python
cimport c_numpy
import numpy as np
import multiprocessing

from cython.parallel cimport parallel
from cython.parallel import prange
cimport openmp

ctypedef unsigned int UINT32
ctypedef unsigned long int UINT64

cdef extern from "stdlib.h":
    void free(void* ptr)
    void* malloc(size_t size)
    void* realloc(void* ptr, size_t size)

cdef extern from "mtm_stats_core.h":
    ctypedef struct SparseBlockArray:
        const UINT32* locs
        const UINT64* array
        UINT32 len
    
    ctypedef struct SparseSetCounts:
        UINT32 i
        UINT32 j
        UINT32 intersection_count
        UINT32 union_count
    
    void compute_base_counts(SparseBlockArray * sba_rows,
                             int chunk_length,
                             int num_rows,
                             UINT32 * totals)
    
    int compute_intersection_and_union_counts(SparseBlockArray * sba_rows,
                                          int chunk_length,
                                          int i,
                                          int num_rows,
                                          SparseSetCounts * sparse_counts,
                                          int cutoff) nogil


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

def cy_mtm_stats(sba_list, chunk_length, cutoff=0):
    '''Run mtm_stats on 64-bit arrays
       Inputs:
        * sba_list: list of sparse block arrays (python format)
                    compressed version of the subset of B connected to each element of A
                    specifically, an list of dictionaries with fields 'array' and 'locs'
        * chunk_length: sba compression parameter
        * cutoff: maximum size of intersection to keep in the output
       
       Returns a numpy structured array with the following fields:
        * i, j: pair of indices into set A (set of interest)
        * intersection_count: number of elements in B that the 
                              A[i] and A[j] share in common
        * union_count: number of elements in B that the 
                       A[i] and A[j] connect to in total
    '''
    cdef int i
    
    num_items = len(sba_list)
    
    # Map the numpy arrays directly to C pointers
    cdef SparseBlockArray * sba_pointer = <SparseBlockArray *> malloc(num_items * sizeof(SparseBlockArray))
    
    for i in range(num_items):
        set_SBA_from_py_dict(sba_pointer, sba_list[i], sba_ind=i)
    
    # Define common vars used below
    cdef int num_threads = multiprocessing.cpu_count()
    cdef int num_items_c = num_items
    cdef int chunk_length_c = chunk_length
    cdef int cutoff_c = cutoff
    
    # Compute the basic counts (bitsum the sba's)
    base_counts = np.zeros(num_items, dtype=np.uint32)
    cdef c_numpy.ndarray base_counts_cn = base_counts
    cdef UINT32 * base_counts_pointer
    base_counts_pointer = <UINT32 *> base_counts_cn.data
    
    compute_base_counts(sba_pointer,
                        chunk_length_c,
                        num_items_c,
                        base_counts_pointer)
    
    # Run compute_intersection_and_union_counts function on the generated pointers
    
    
    # Set up results buffer and pointers
    # This section:
    # * Makes a num_threads by num_items numpy array
    # * Makes a C double pointer (array-of-arrays) of SparseSetCounts
    # * Sets the pointers to each of the numpy sub-arrays (each is num_items)
    # Each thread then gets it's own num_items length buffer to store its result
    sparse_counts = np.zeros((num_threads, num_items), dtype=SPARSE_COUNTS_DTYPE) # Make sure each thread uses separate memory
    cdef c_numpy.ndarray sparse_counts_cn
    cdef SparseSetCounts ** sparse_counts_pointer_arr
    sparse_counts_pointer_arr = <SparseSetCounts **> malloc(num_threads * sizeof(SparseSetCounts *))
    for i in range(num_threads):
        sparse_counts_cn = sparse_counts[i]
        sparse_counts_pointer_arr[i] = <SparseSetCounts*> sparse_counts_cn.data
    
    cdef int num_sparse_counts
    cdef int thread_number
    sparse_counts_list = [None] * num_items
    for i in prange(num_items_c, nogil=True, chunksize=1, num_threads=num_threads, schedule='static'):
        thread_number = openmp.omp_get_thread_num()
        num_sparse_counts = compute_intersection_and_union_counts(
                                sba_pointer,
                                chunk_length_c,
                                i,
                                num_items_c,
                                sparse_counts_pointer_arr[thread_number],
                                cutoff_c
                            )
        
        with gil:
            sparse_counts_list[i] = sparse_counts[thread_number][:num_sparse_counts]

    # Collect the results
    all_sparse_counts = np.concatenate(sparse_counts_list)
    
    
    # Return the results from the two sections
    return base_counts, all_sparse_counts
