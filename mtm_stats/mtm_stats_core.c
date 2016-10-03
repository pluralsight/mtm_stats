#include <stdbool.h>
#include "mtm_stats_core.h"

static UINT32 bit_sum(CONSTANT UINT64* bit_array, int length) {
//Given a 64-bit array, sum all the individual to corresponding binary and sum.
    int i;
    UINT32 sum = 0;
    for(i = 0; i < length; i++)
        sum += POPCOUNT(bit_array[i]);
    return sum;
}

static UINT32 bit_sum_or(CONSTANT UINT64* a_array, CONSTANT UINT64* b_array, int length) {
// Sum the bitwise OR for each item
    int i;
    UINT32 sum = 0;
    for(i = 0; i < length; i++) {
        sum += POPCOUNT(a_array[i] | b_array[i]);
    }
    return sum;
}

static UINT32 bit_sum_and(CONSTANT UINT64* a_array, CONSTANT UINT64* b_array, int length) {
// Sum the bitwise AND for each item
    int i;
    UINT32 sum = 0;
    for(i = 0; i < length; i++) {
        sum += POPCOUNT(a_array[i] & b_array[i]);
    }
    return sum;
}

static UINT32 sparse_bit_sum(SparseBlockArray a, int chunk_length) {
    return bit_sum(a.array, chunk_length * a.len);
}

static UINT32 sparse_bit_sum_or(SparseBlockArray a, SparseBlockArray b, int chunk_length) {
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

static UINT32 sparse_bit_sum_and(SparseBlockArray a, SparseBlockArray b, int chunk_length) {
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

void compute_connection_counts(SparseBlockArray * sba_rows, int chunk_length, int num_rows, int * totals) {
//Compute the bit_sum of each SBA (row) in an array
//Store the result in "totals" (pre-allocated with length "num_rows")
    int i;
    for(i=0;i<num_rows;i++) {
        totals[i] = sparse_bit_sum(sba_rows[i], chunk_length);
    }
}

bool compute_intersection_and union_count(SparseBlockArray * sba_rows, int chunk_length, int i, int j, SparseSetCount * sparse_count, int cutoff) {
//Compute the counts (intersection and union) between two rows in an array of SBA's
//if the intersection is greater than the cutoff, return the intersection and the union in the sparse_count and return true
//otherwise return false (result not be saved)
    UINT32 intersection = sparse_bit_sum_and(sba_rows[i], sba_rows[j]);
    if(intersection <= cutoff) {
        return false;
    } else {
        sparse_count -> i = i;
        sparse_count -> j = j;
        sparse_count -> intersection = intersection;
        sparse_count -> union = sparse_bit_sum_or(sba_rows[i], sba_rows[j]);
        return true;
    }
}

int compute_intersection_and union_counts(SparseBlockArray * sba_rows, int chunk_length, int i, int num_rows, SparseSetCounts * sparse_counts, int cutoff) {
//Compute SparseSetCounts of all rows larger than ia with ia
//sparse_counts must be pre-allocated with a length of num_rows
    int j;
    bool result;
    int num_sparse_counts = 0;
    for(j = j+1; j < num_rows; j++) {
        result = compute_count(sba_rows, chunk_length, i, j,  &sparse_counts[num_sparse_counts], cutoff)
        if(result) {
            num_sparse_counts++;
        }
    }
    return num_sparse_counts;
}

