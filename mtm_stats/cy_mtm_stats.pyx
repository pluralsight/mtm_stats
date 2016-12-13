import numpy as np
cimport numpy as np

import multiprocessing
from cython.parallel cimport parallel
from cython.parallel import prange
cimport openmp

ctypedef unsigned int UINT32
ctypedef unsigned long int UINT64

cdef extern from "stdlib.h":
    ctypedef int size_t
    void free(void* ptr)
    void* malloc(size_t size)
    void* realloc(void* ptr, size_t size)

cdef extern from "mtm_stats_core.h":
    ctypedef struct SparseBlockArray:
        const UINT32* locs
        const UINT64* array
        UINT32 len
    
    ctypedef struct IntersectionCount:
        UINT32 i
        UINT32 j
        UINT32 intersection_count
    
    void compute_counts(SparseBlockArray * sba_rows,
                        int chunk_length,
                        int num_rows,
                        UINT32 * counts)
    
    int compute_intersection_counts(SparseBlockArray * sba_rows,
                                    int chunk_length,
                                    int i,
                                    int num_rows,
                                    IntersectionCount * intersection_counts,
                                    int cutoff) nogil


INTERSECTION_COUNTS_DTYPE = [('i', np.uint32),
                             ('j', np.uint32),
                             ('intersection_count', np.uint32)]

cdef void set_SBA_from_py_dict(SparseBlockArray * sba, input_dict, sba_ind=0):
    '''Takes a python dictionary of arrays and fill a SparseBlockArray
       The expected input format for the input_dict is:
       {'locs': <numpy array of uint32>,
        'array': <numpy array of uint64>}
       '''
    cdef np.ndarray locs_cn = input_dict['locs']
    cdef np.ndarray array_cn = input_dict['array']
    
    sba[sba_ind].locs = <const UINT32 *> locs_cn.data
    sba[sba_ind].array = <const UINT64 *> array_cn.data
    sba[sba_ind].len = len(locs_cn)

def cy_compute_counts(sba_list, chunk_length, cutoff=0):
    '''Wrapper around compute_counts
       Inputs:
        * sba_list: list of sparse block arrays (python format)
                    compressed version of the subset of B connected to each element of A
                    specifically, an list of dictionaries with fields 'array' and 'locs'
        * chunk_length: sba compression parameter
        * cutoff: maximum size of intersection to keep in the output
       
       Returns a numpy array (uint32) with the count for each sba_list
    '''
    
    cdef int i
    
    num_items = len(sba_list)
    
    cdef int num_items_c = num_items
    cdef int chunk_length_c = chunk_length
    
    # Map the numpy arrays directly to C pointers
    cdef SparseBlockArray * sba_pointer = <SparseBlockArray *> malloc(num_items * sizeof(SparseBlockArray))
    
    for i in range(num_items):
        set_SBA_from_py_dict(sba_pointer, sba_list[i], sba_ind=i)
    
    # Compute the counts (bitsum the sba's)
    counts = np.zeros(num_items, dtype=np.uint32)
    cdef np.ndarray counts_cn = counts
    cdef UINT32 * counts_pointer
    counts_pointer = <UINT32 *> counts_cn.data
    
    compute_counts(sba_pointer,
                   chunk_length_c,
                   num_items_c,
                   counts_pointer)
    
    return counts

def cy_compute_intersection_counts(sba_list, chunk_length, cutoff=0):
    '''Wrapper around compute_intersection_counts
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
    '''
    
    cdef int i
    
    num_items = len(sba_list)
    
    cdef int num_threads = multiprocessing.cpu_count()
    cdef int num_items_c = num_items
    cdef int chunk_length_c = chunk_length
    cdef int cutoff_c = cutoff
    
    # Map the numpy arrays directly to C pointers
    cdef SparseBlockArray * sba_pointer = <SparseBlockArray *> malloc(num_items * sizeof(SparseBlockArray))
    
    for i in range(num_items):
        set_SBA_from_py_dict(sba_pointer, sba_list[i], sba_ind=i)
    
    # Run compute_intersection_counts on the generated pointers:
    
    # Set up results buffer and pointers
    # This section:
    # * Makes a num_threads by num_items numpy array
    # * Makes a C double pointer (array-of-arrays) of IntersectionCounts
    # * Sets the pointers to each of the numpy sub-arrays (each is num_items)
    # Each thread then gets it's own num_items length buffer to store its result
    intersection_counts_tmp_arr = np.zeros((num_threads, num_items),        # Make sure each thread uses separate memory
                                           dtype=INTERSECTION_COUNTS_DTYPE)
    cdef np.ndarray intersection_counts_cn
    cdef IntersectionCount ** intersection_counts_pointer_arr
    intersection_counts_pointer_arr = <IntersectionCount **> malloc(num_threads * sizeof(IntersectionCount *))
    for i in range(num_threads):
        intersection_counts_cn = intersection_counts_tmp_arr[i]
        intersection_counts_pointer_arr[i] = <IntersectionCount*> intersection_counts_cn.data
    
    cdef int num_intersection_counts
    cdef int thread_number
    intersection_counts_list = [None] * num_items
    for i in prange(num_items_c, nogil=True, chunksize=1, num_threads=num_threads, schedule='static'):
        thread_number = openmp.omp_get_thread_num()
        num_intersection_counts = compute_intersection_counts(sba_pointer,
                                                              chunk_length_c,
                                                              i,
                                                              num_items_c,
                                                              intersection_counts_pointer_arr[thread_number],
                                                              cutoff_c)
        with gil:
            intersection_counts_list[i] = np.array(intersection_counts_tmp_arr[thread_number][:num_intersection_counts])

    # Collect the results and return
    intersection_counts = np.concatenate(intersection_counts_list)
    return intersection_counts

def cy_mtm_stats(sba_list, chunk_length, cutoff=0):
    '''Run mtm_stats on 64-bit arrays
       Inputs:
        * sba_list: list of sparse block arrays (python format)
                    compressed version of the subset of B connected to each element of A
                    specifically, an list of dictionaries with fields 'array' and 'locs'
        * chunk_length: sba compression parameter
        * cutoff: maximum size of intersection to keep in the output
       
       Returns two numpy arrays:
       A uint32 array with the basic counts
       A numpy structured array with the following fields:
        * i, j: pair of indices into set A (set of interest)
        * intersection_count: number of elements in B that the 
                              A[i] and A[j] share in common
    '''
    counts = cy_compute_counts(sba_list, chunk_length, cutoff)
    intersection_counts = cy_compute_intersection_counts(sba_list, chunk_length, cutoff)

    # Return the results from the two sections
    return counts, intersection_counts
