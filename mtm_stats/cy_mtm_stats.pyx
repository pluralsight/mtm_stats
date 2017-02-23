import numpy as np
cimport numpy as np

import multiprocessing
from cython.parallel cimport parallel
from cython.parallel import prange
cimport openmp

ctypedef unsigned int UINT32
ctypedef unsigned long int UINT64
ctypedef int INT32

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
                                    int start_j,
                                    int num_rows,
                                    IntersectionCount * intersection_counts,
                                    int cutoff) nogil

    void compute_counts_dense_input(UINT64 * rows_arr,
                                    int chunk_length,
                                    int num_rows,
                                    UINT32 * counts)
                
    int compute_intersection_counts_dense_input(UINT64 * rows_arr,
                                                int chunk_length,
                                                int i,
                                                int start_j,
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

def cy_compute_counts(sba_list, chunk_length):
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

def cy_compute_intersection_counts(sba_list, chunk_length, indices_a=None, cutoff=0, start_j=0, upper_only=True):
    '''Wrapper around compute_intersection_counts
       Inputs:
        * sba_list: list of sparse block arrays (python format)
                    compressed version of the subset of B connected to each element of A
                    specifically, an list of dictionaries with fields 'array' and 'locs'
        * chunk_length: sba compression parameter
        * cutoff: maximum size of intersection to keep in the output
        * start_j: an offset to apply on comparison
          (skip comparing against values less than start_j)
       
       Returns a list of numpy structured arrays with the following fields:
        * i, j: pair of indices into set A (set of interest)
        * intersection_count: number of elements in B that the 
                              A[i] and A[j] share in common
    '''
    
    cdef int i, ii
    
    num_items = len(sba_list)
    
    indices_a = np.asanyarray((np.arange(num_items)
                               if indices_a is None else
                               indices_a),
                              dtype=np.int32)
    
    num_a = len(indices_a)
    
    intersection_counts_list = [None] * num_a
    
    cdef int num_threads = multiprocessing.cpu_count()
    cdef int num_items_c = num_items
    cdef int chunk_length_c = chunk_length
    cdef int cutoff_c = cutoff
    cdef int start_j_c = start_j
    cdef int upper_only_c = upper_only
    
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
    
    cdef np.ndarray indices_a_cn = indices_a
    cdef INT32 * indices_a_pointer
    indices_a_pointer = <INT32 *> indices_a_cn.data
    
    for ii in prange(num_a, nogil=True, chunksize=1, num_threads=num_threads, schedule='static'):
        i = indices_a_pointer[ii] # add a layer of indirection, but should still be fast
        thread_number = openmp.omp_get_thread_num()
        num_intersection_counts = compute_intersection_counts(sba_pointer,
                                                              chunk_length_c,
                                                              i,
                                                              i+1 if upper_only_c and start_j_c <= i else start_j_c,
                                                              num_items_c,
                                                              intersection_counts_pointer_arr[thread_number],
                                                              cutoff_c)
        with gil:
            intersection_counts_list[ii] = np.array(intersection_counts_tmp_arr[thread_number][:num_intersection_counts])

    return intersection_counts_list

def cy_compute_counts_dense_input(rows_arr):
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
    
    num_items, chunk_length = rows_arr.shape
    
    cdef int num_items_c = num_items
    cdef int chunk_length_c = chunk_length
    
    # Map the numpy arrays directly to C pointers
    cdef np.ndarray rows_cn
    rows_cn = rows_arr
    cdef UINT64 * rows_pointer = <const UINT64 *> rows_cn.data
    
    # Compute the counts (bitsum the rows)
    counts = np.zeros(num_items, dtype=np.uint32)
    cdef np.ndarray counts_cn = counts
    cdef UINT32 * counts_pointer
    counts_pointer = <UINT32 *> counts_cn.data
    
    compute_counts_dense_input(rows_pointer,
                               chunk_length_c,
                               num_items_c,
                               counts_pointer)
    
    return counts

def cy_compute_intersection_counts_dense_input(rows_arr, indices_a=None, cutoff=0, start_j=0, upper_only=True):
    '''Wrapper around compute_intersection_counts_dense_input
       Inputs:
        * rows_arr: array of uint64 values with shape (num_rows, chunk_length)
        * indices_a: parameter to allow only running computing intersections
                     of certain values against the rest of the values
                     (with default None, compute all indices against all others)
        * cutoff: maximum size of intersection to keep in the output
        * start_j: an offset to apply on comparison
          (skip comparing against values less than start_j)
       
       Returns a list of numpy structured arrays with the following fields:
        * i, j: pair of indices into set A (set of interest)
        * intersection_count: number of elements in B that the 
                              A[i] and A[j] share in common
    '''
    
    cdef int i, ii
    
    num_items, chunk_length = rows_arr.shape
    
    indices_a = np.asanyarray((np.arange(num_items)
                               if indices_a is None else
                               indices_a),
                              dtype=np.int32)
    
    num_a = len(indices_a)
    
    intersection_counts_list = [None] * num_a
    
    cdef int num_threads = multiprocessing.cpu_count()
    cdef int num_items_c = num_items
    cdef int chunk_length_c = chunk_length
    cdef int cutoff_c = cutoff
    cdef int start_j_c = start_j
    cdef int upper_only_c = upper_only
    
    # Map the numpy arrays directly to C pointers
    cdef np.ndarray rows_cn
    rows_cn = rows_arr
    cdef UINT64 * rows_pointer = <const UINT64 *> rows_cn.data
    
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
    
    cdef np.ndarray indices_a_cn = indices_a
    cdef INT32 * indices_a_pointer
    indices_a_pointer = <INT32 *> indices_a_cn.data
    
    for ii in prange(num_a, nogil=True, chunksize=1, num_threads=num_threads, schedule='static'):
        i = indices_a_pointer[ii] # add a layer of indirection, but should still be fast
        thread_number = openmp.omp_get_thread_num()
        num_intersection_counts = compute_intersection_counts_dense_input(rows_pointer,
                                                                          chunk_length_c,
                                                                          i,
                                                                          i+1 if upper_only_c and start_j_c <= i else start_j_c,
                                                                          num_items_c,
                                                                          intersection_counts_pointer_arr[thread_number],
                                                                          cutoff_c)
        with gil:
            intersection_counts_list[ii] = np.array(intersection_counts_tmp_arr[thread_number][:num_intersection_counts])

    return intersection_counts_list


def cy_mtm_stats(sba_list, chunk_length, indices_a=None, cutoff=0, start_j=0, upper_only=True):
    '''Run mtm_stats on 64-bit arrays
       Inputs:
        * sba_list: list of sparse block arrays (python format)
                    compressed version of the subset of B connected to each element of A
                    specifically, an list of dictionaries with fields 'array' and 'locs'
        * chunk_length: sba compression parameter
        * cutoff: maximum size of intersection to keep in the output
       
       Returns two items:
       base_counts:
           A uint32 array with the basic counts
       
       intersection_counts_list:
           A list of numpy structured arrays with the following fields:
             * i, j: pair of indices into set A (set of interest)
             * intersection_count: number of elements in B that the 
                                   A[i] and A[j] share in common
    '''
    base_counts = cy_compute_counts(sba_list, chunk_length)
    intersection_counts_list = cy_compute_intersection_counts(sba_list, chunk_length, indices_a, cutoff, start_j, upper_only)

    # Return the results from the two sections
    return base_counts, intersection_counts_list

def cy_mtm_stats_dense_input(rows_arr, indices_a=None, cutoff=0, start_j=0, upper_only=True):
    '''Run mtm_stats on 64-bit arrays
       Inputs:
        * sba_list: list of sparse block arrays (python format)
                    compressed version of the subset of B connected to each element of A
                    specifically, an list of dictionaries with fields 'array' and 'locs'
        * chunk_length: sba compression parameter
        * cutoff: maximum size of intersection to keep in the output
       
       Returns two items:
       base_counts:
           A uint32 array with the basic counts
       
       intersection_counts_list:
           A list of numpy structured arrays with the following fields:
             * i, j: pair of indices into set A (set of interest)
             * intersection_count: number of elements in B that the 
                                   A[i] and A[j] share in common
    '''
    base_counts = cy_compute_counts_dense_input(rows_arr)
    intersection_counts_list = cy_compute_intersection_counts_dense_input(rows_arr, indices_a, cutoff, start_j, upper_only)

    # Return the results from the two sections
    return base_counts, intersection_counts_list
