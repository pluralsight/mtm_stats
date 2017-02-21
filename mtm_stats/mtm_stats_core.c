#include <stdbool.h>
#include "mtm_stats_core.h"

// Create defines to make code easier to read below
// DAT_X automatically gets a pointer to the i_x'th chunk of dat_x
#define DAT_A ( &a.array[i_a * chunk_length] )
#define DAT_B ( &b.array[i_b * chunk_length] )

static UINT32 bit_sum(CONSTANT UINT64* bit_array,
                      int length) {
//Given a 64-bit array, sum all the individual to corresponding binary and sum.
    int i;
    UINT32 sum = 0;
    for(i = 0; i < length; i++)
        sum += POPCOUNT(bit_array[i]);
    return sum;
}

static UINT32 bit_sum_or(CONSTANT UINT64* a_array,
                         CONSTANT UINT64* b_array,
                         int length) {
// Sum the bitwise OR for each item
    int i;
    UINT32 sum = 0;
    for(i = 0; i < length; i++) {
        sum += POPCOUNT(a_array[i] | b_array[i]);
    }
    return sum;
}

static UINT32 bit_sum_and(CONSTANT UINT64* a_array,
                          CONSTANT UINT64* b_array,
                          int length) {
// Sum the bitwise AND for each item
    int i;
    UINT32 sum = 0;
    for(i = 0; i < length; i++) {
        sum += POPCOUNT(a_array[i] & b_array[i]);
    }
    return sum;
}

static UINT32 sparse_bit_sum(SparseBlockArray a,
                             int chunk_length) {
    return bit_sum(a.array, chunk_length * a.len);
}

static UINT32 sparse_bit_sum_or(SparseBlockArray a,
                                SparseBlockArray b,
                                int chunk_length) {
// Same as bit_sum_or but over sparse block arrays
// Each array is an ordered list of locations in the full array (loc)
// and a list of chunks of UINT64 data, each of size chunk_length (dat)
    int i;
    int i_a = 0;
    int i_b = 0;
    UINT32 sum = 0;
    for(i=0; i<(a.len + b.len); i++) {
        //First check bounds, then check whether a or b has greater value
        if(i_a >= a.len) {
            sum += bit_sum(DAT_B, chunk_length);
            i_b++;
        } else if(i_b >= b.len) {
            sum += bit_sum(DAT_A, chunk_length);
            i_a++;
        } else if(a.locs[i_a] < b.locs[i_b]) {
            sum += bit_sum(DAT_A, chunk_length);
            i_a++;
        } else if(b.locs[i_b] < a.locs[i_a]) {
            sum += bit_sum(DAT_B, chunk_length);
            i_b++;
        } else {
            sum += bit_sum_or(DAT_A, DAT_B, chunk_length);
            i_a++; i_b++; i++;
        }
    }
    return sum;
}

static UINT32 sparse_bit_sum_and(SparseBlockArray a,
                                 SparseBlockArray b,
                                 int chunk_length) {
// Same as bit_sum_or but over sparse block arrays
// Each array is an ordered list of locations in the full array (loc)
// and a list of chunks of UINT64 data, each of size chunk_length (dat)
// This operation is especially simple because only matching chunks contribute to the sum
    int i;
    int i_a = 0;
    int i_b = 0;
    UINT32 sum = 0;
    for(i=0; i<(a.len + b.len); i++) {
        //First check bounds, then check whether a or b has greater value
        if(i_a >= a.len || i_b >= b.len) {
            break;
        } else if(b.locs[i_b] < a.locs[i_a]) {
            i_b++;
        } else if(a.locs[i_a] < b.locs[i_b]) {
            i_a++;
        } else {
            sum += bit_sum_and(DAT_A, DAT_B, chunk_length);
            i_a++; i_b++; i++;
        }
    }
    return sum;
}

void compute_counts(SparseBlockArray * sba_rows,
                    int chunk_length,
                    int num_rows,
                    UINT32 * counts) {
//Compute the bit_sum of each SBA (row) in an array
//Store the result in "counts" (pre-allocated with length "num_rows")
    int i;
    for(i=0;i<num_rows;i++) {
        counts[i] = sparse_bit_sum(sba_rows[i], chunk_length);
    }
}

bool compute_intersection_count(SparseBlockArray * sba_rows,
                                int chunk_length,
                                int i,
                                int j,
                                IntersectionCount * intersection_count_ptr,
                                int cutoff) {
//Compute the counts (intersection and union) between two rows in an array of SBA's
//if the intersection is greater than the cutoff, return the intersection and the union in the intersection_counts and return true
//otherwise return false (result not be saved)
    UINT32 count = sparse_bit_sum_and(sba_rows[i],
                                      sba_rows[j],
                                      chunk_length);
    if(count <= cutoff) {
        return false;
    } else {
        intersection_count_ptr -> i = i;
        intersection_count_ptr -> j = j;
        intersection_count_ptr -> intersection_count = count;
        return true;
    }
}

int compute_intersection_counts(SparseBlockArray * sba_rows,
                                int chunk_length,
                                int i,
                                int start_j,
                                int num_rows,
                                IntersectionCount * intersection_counts,
                                int cutoff) {
//Compute IntersectionCount of all rows larger than ia with ia
//intersection_counts must be pre-allocated with a length of num_rows
//start_j should be (i + 1) under normal circumstances
    int j;
    bool result;
    int num_intersection_counts = 0;
    for(j = start_j; j < num_rows; j++) {
        if(i == j) { continue; } // skip computing the row with itself
        result = compute_intersection_count(sba_rows,
                                            chunk_length,
                                            i,
                                            j,
                                            &intersection_counts[num_intersection_counts],
                                            cutoff);
        if(result) {
            num_intersection_counts++;
        }
    }
    return num_intersection_counts;
}

////////////////////////////////////////////////////////////////////////
// Add 3 new functions that take dense arrays (simple UINT64 pointers)
// instead of SparseBlockArray pointers.
// These still allow for chunk_length to be used
////////////////////////////////////////////////////////////////////////

void compute_counts_dense_input(UINT64 * rows_arr,
                                int chunk_length,
                                int num_rows,
                                UINT32 * counts) {
//Compute the bit_sum of each SBA (row) in an array
//Store the result in "counts" (pre-allocated with length "num_rows")
    int i;
    for(i=0;i<num_rows;i++) {
        counts[i] = bit_sum(&rows_arr[i * chunk_length],
                            chunk_length);
    }
}

bool compute_intersection_count_dense_input(UINT64 * rows_arr,
                                            int chunk_length,
                                            int i,
                                            int j,
                                            IntersectionCount * intersection_count_ptr,
                                            int cutoff) {
//Compute the counts (intersection and union) between two rows in an array of SBA's
//if the intersection is greater than the cutoff, return the intersection and the union in the intersection_counts and return true
//otherwise return false (result not be saved)
    UINT32 count = bit_sum_and(&rows_arr[i * chunk_length],
                               &rows_arr[j * chunk_length],
                               chunk_length);
    if(count <= cutoff) {
        return false;
    } else {
        intersection_count_ptr -> i = i;
        intersection_count_ptr -> j = j;
        intersection_count_ptr -> intersection_count = count;
        return true;
    }
}

int compute_intersection_counts_dense_input(UINT64 * rows_arr,
                                            int chunk_length,
                                            int i,
                                            int start_j,
                                            int num_rows,
                                            IntersectionCount * intersection_counts,
                                            int cutoff) {
//Compute IntersectionCount of all rows larger than ia with ia
//intersection_counts must be pre-allocated with a length of num_rows
//start_j should be (i + 1) under normal circumstances
    int j;
    bool result;
    int num_intersection_counts = 0;
    for(j = start_j; j < num_rows; j++) {
        if(i == j) { continue; } // skip computing the row with itself
        result = compute_intersection_count_dense_input(rows_arr,
                                                        chunk_length,
                                                        i,
                                                        j,
                                                        &intersection_counts[num_intersection_counts],
                                                        cutoff);
        if(result) {
            num_intersection_counts++;
        }
    }
    return num_intersection_counts;
}

