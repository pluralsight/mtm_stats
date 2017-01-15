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
} IntersectionCount;

void compute_counts(SparseBlockArray * sba_rows,
                    int chunk_length,
                    int num_rows,
                    UINT32 * counts);

int compute_intersection_counts(SparseBlockArray * sba_rows,
                                int chunk_length,
                                int i,
                                int start_j,
                                int num_rows,
                                IntersectionCount * intersection_counts,
                                int cutoff);

void compute_counts_dense_input(UINT64 * rows_arr,
                                int chunk_length,
                                int num_rows,
                                UINT32 * counts);

int compute_intersection_counts_dense_input(UINT64 * rows_arr,
                                            int chunk_length,
                                            int i,
                                            int start_j,
                                            int num_rows,
                                            IntersectionCount * intersection_counts,
                                            int cutoff);

#endif
