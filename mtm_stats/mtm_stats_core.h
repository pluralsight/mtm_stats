#ifndef MTM_STATS_CORE_H
#define MTM_STATS_CORE_H

#define CONSTANT const
#define POPCOUNT __builtin_popcountl

//#define GLOBAL
//#define KERNEL
//#define GET_GLOBAL_ID 0

//For OpenCL, these needs to be:
// #define KERNEL __kernel
// #define CONSTANT __constant
// #define GLOBAL __global
// #define POPCOUNT popcount
// #define GET_GLOBAL_ID get_global_id(0)
// #define GENERATED=GENERATED_STRING,


typedef unsigned int UINT32;
typedef unsigned long int UINT64;

typedef struct {
    CONSTANT UINT32* locs;
    CONSTANT UINT64* array;
    UINT32 len;
} SparseBlockArray;

typedef struct {
    UINT32 i;
    UINT32 j;
    UINT32 intersection_count;
    UINT32 union_count;
} SparseSetCounts;

void compute_base_counts(SparseBlockArray * sba_rows,
                         int chunk_length,
                         int num_rows,
                         UINT32 * totals);

int compute_intersection_and_union_counts(SparseBlockArray * sba_rows,
                                          int chunk_length,
                                          int i,
                                          int num_rows,
                                          SparseSetCounts * sparse_counts,
                                          int cutoff);

#endif
