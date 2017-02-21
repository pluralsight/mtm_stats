'''Test it all'''
from __future__ import division
from builtins import range

import numpy as np
import mtm_stats
from mtm_stats.testing_utils import (generate_test_set, run_timing_test,
                                     naive_counts, naivest_counts, is_naive_same)


TEST_SET_1 = [('a1', 'b1'),
              ('a1', 'b2'),
              ('a1', 'b3'),
              ('a2', 'b1'),
              ('a2', 'b2'),
              ('a3', 'b3'),
              ('a4', 'b9'),]

def test_mtm_stats_1():
    bcd, scd = mtm_stats.mtm_stats(TEST_SET_1)
    assert bcd == {'a1': 3, 'a3': 1, 'a2': 2, 'a4': 1}
    assert scd == {('a1', 'a2'): (2, 3), ('a1', 'a3'): (1, 3)}

def test_mtm_stats_2():
    bcd, scd = mtm_stats.mtm_stats(TEST_SET_1)
    assert bcd == {'a1': 3, 'a3': 1, 'a2': 2, 'a4': 1}
    assert scd == {('a1', 'a2'): (2, 3), ('a1', 'a3'): (1, 3)}
    assert (bcd, scd) == naive_counts(TEST_SET_1)
    assert (bcd, scd) == naivest_counts(TEST_SET_1)

def test_mtm_stats_iterator_1():
    assert mtm_stats.mtm_stats(TEST_SET_1) == mtm_stats.mtm_stats_from_iterator(TEST_SET_1, 2)

def test_mtm_stats_iterator_2():
    test_set = generate_test_set(sizeA=143,
                                 sizeB=157,
                                 num_connections=20400)
    assert mtm_stats.mtm_stats(test_set) == mtm_stats.mtm_stats_from_iterator(test_set, 10)

def test_mtm_stats_upper_only_False_1():
    test_set = generate_test_set(sizeA=143,
                                 sizeB=157,
                                 num_connections=20400)
    bch, half = mtm_stats.mtm_stats(test_set)
    bcf, full = mtm_stats.mtm_stats(test_set, upper_only=False)
    assert bch == bcf
    assert 2 * len(half) == len(full)
    assert half == {k: v for k, v in full.items()
                         if k[0] < k[1]}

def test_mtm_stats_sizeA_10_sizeB_20_num_connections_20():
    assert is_naive_same(generate_test_set(sizeA=10,
                                           sizeB=20,
                                           num_connections=20))


def test_mtm_stats_sizeA_10_sizeB_10_num_connections_100():
    assert is_naive_same(generate_test_set(sizeA=10,
                                           sizeB=10,
                                           num_connections=100))

def test_mtm_stats_sizeA_100_sizeB_10000_num_connections_10000():
    assert is_naive_same(generate_test_set(sizeA=100,
                                           sizeB=10000,
                                           num_connections=10000))

def test_mtm_stats_sizeA_1000_sizeB_100000_num_connections_50000():
    assert is_naive_same(generate_test_set(sizeA=1000,
                                           sizeB=100000,
                                           num_connections=50000),
                        print_time=True)

def test_get_Jaccard_index_1():
    bcd, ji = mtm_stats.get_Jaccard_index(TEST_SET_1)
    assert bcd == {'a1': 3, 'a3': 1, 'a2': 2, 'a4': 1}
    assert ji == {('a1', 'a2'): 2/3, ('a1', 'a3'): 1/3}

def test_get_mtm_dense_vs_sparse_1():
    connections = generate_test_set(sizeA=100,
                                    sizeB=10000,
                                    num_connections=10000)
    assert (mtm_stats.mtm_stats(connections, dense_input=False, indices_a=range(10)) == 
            mtm_stats.mtm_stats(connections, dense_input=True, indices_a=range(10)))

def test_get_mtm_with_indices_a_dense_vs_sparse_1():
    connections = generate_test_set(sizeA=100,
                                    sizeB=10000,
                                    num_connections=10000)
    assert (mtm_stats.mtm_stats(connections, dense_input=False, indices_a=range(10)) ==
            mtm_stats.mtm_stats(connections, dense_input=True, indices_a=range(10)))





def performance_test_sizeA_100_sizeB_10000_num_connections_10000():
    gt, mt = run_timing_test(sizeA=100,
                             sizeB=10000,
                             num_connections=10000)
    assert mt < 1

def performance_test_sizeA_10000_sizeB_1000000_num_connections_1000000():
    gt, mt = run_timing_test(sizeA=10000,
                             sizeB=1000000,
                             num_connections=1000000)
    assert mt < 100, 'mtm_stats is too slow'
    # Uses about 0.2GB, 20s (on my machine)


def performance_test_sizeA_10000_sizeB_10000000_num_connections_1000000():
    gt, mt = run_timing_test(sizeA=10000,
                             sizeB=10000000,
                             num_connections=1000000)
    assert mt < 100, 'mtm_stats is too slow'
    # Uses about 0.8GB, 19s (on my machine)

def performance_test_sizeA_5000_sizeB_20000000_num_connections_1000000():
    gt, mt = run_timing_test(sizeA=5000,
                             sizeB=20000000,
                             num_connections=1000000)
    assert mt < 100, 'mtm_stats is too slow'
    # Uses about 1.5GB, 13s (on my machine)

def performance_test_sizeA_2000_sizeB_50000000_num_connections_1000000():
    gt, mt = run_timing_test(sizeA=2000,
                             sizeB=50000000,
                             num_connections=1000000)
    assert mt < 100, 'mtm_stats is too slow'
    # Uses about 3.5GB, 8s (on my machine)

def performance_test_sizeA_5000_sizeB_10000000_num_connections_2000000():
    gt, mt = run_timing_test(sizeA=5000,
                             sizeB=10000000,
                             num_connections=2000000)
    assert mt < 100, 'mtm_stats is too slow'
    # Uses about 0.9GB, 29s (on my machine)

def performance_test_sizeA_2000_sizeB_10000000_num_connections_5000000():
    gt, mt = run_timing_test(sizeA=2000,
                             sizeB=10000000,
                             num_connections=5000000)
    assert mt < 100, 'mtm_stats is too slow'
    # Uses about 1.1GB, 43s (on my machine)

def performance_test_sizeA_1000_sizeB_10000000_num_connections_10000000():
    gt, mt = run_timing_test(sizeA=1000,
                             sizeB=10000000,
                             num_connections=10000000)
    assert mt < 100, 'mtm_stats is too slow'
    # Uses about 15GB, 54s (on my machine)

def performance_test_sizeA_10000_sizeB_20000000_num_connections_1000000():
    '''Pushing it until it breaks :) '''
    gt, mt = run_timing_test(sizeA=10000,
                             sizeB=20000000,
                             num_connections=1000000)
    # Uses about 1.5GB, 21s (on my machine)

def performance_test_sizeA_20000_sizeB_10000000_num_connections_1000000():
    '''Pushing it until it breaks :) '''
    gt, mt = run_timing_test(sizeA=20000,
                             sizeB=10000000,
                             num_connections=1000000)
    # Uses about 0.8GB, 36s (on my machine)

def performance_test_sizeA_40000_sizeB_10000000_num_connections_1000000():
    '''Pushing it until it breaks :) '''
    gt, mt = run_timing_test(sizeA=40000,
                             sizeB=10000000,
                             num_connections=1000000)
    # Uses about 0.8GB, 62s (on my machine)

def performance_test_sizeA_10000_sizeB_10000000_num_connections_2000000():
    '''Pushing it until it breaks :) '''
    gt, mt = run_timing_test(sizeA=10000,
                             sizeB=10000000,
                             num_connections=2000000)
    # Uses about 0.9GB, 40s (on my machine)

def performance_test_sizeA_10000_sizeB_10000000_num_connections_10000000():
    '''Pushing it until it breaks :) '''
    gt, mt = run_timing_test(sizeA=10000,
                             sizeB=10000000,
                             num_connections=10000000)
    # Uses about 1.6GB, 245s (on my machine)

def performance_test_sizeA_80000_sizeB_10000000_num_connections_10000000():
    '''Pushing it until it breaks :) '''
    gt, mt = run_timing_test(sizeA=80000,
                             sizeB=10000000,
                             num_connections=10000000)
    # Yeah, so it broke -- not enough memory to even make the binary input!
    # This issue actually could be worked around at the cost of some setup
    # speed by looping through the array multiple times to generate subsets
    # and compressing them in smaller pieces
    # Update: the new sba conversion function doesn't crash
    # Uses about 1.7GB, 1078s (on my machine)

def performance_test_sizeA_10000_sizeB_10000000_num_connections_100000000():
    '''Pushing it until it breaks :) '''
    gt, mt = run_timing_test(sizeA=10000,
                             sizeB=10000000,
                             num_connections=100000000)
    # Uses about 10.0GB, 2619s (on my machine)

def performance_test_sizeA_10000_sizeB_20000000_num_connections_1000000_chunk_length_64_2_cutoff_0():
    '''Pushing it until it breaks :) '''
    gt, mt = run_timing_test(sizeA=10000,
                             sizeB=20000000,
                             num_connections=1000000,
                             chunk_length_64=2,
                             cutoff=0)
    # Uses about 1.5GB, 21s (on my machine)


if __name__ == '__main__':
    test_mtm_stats_1()
    test_mtm_stats_2
    test_mtm_stats_iterator_1()
    test_mtm_stats_iterator_2()
    test_mtm_stats_upper_only_False_1()
    test_mtm_stats_sizeA_10_sizeB_20_num_connections_20()
    test_mtm_stats_sizeA_10_sizeB_10_num_connections_100()
    test_mtm_stats_sizeA_100_sizeB_10000_num_connections_10000()
    test_mtm_stats_sizeA_1000_sizeB_100000_num_connections_50000()
    test_get_Jaccard_index_1()
    test_get_mtm_dense_vs_sparse_1()
    test_get_mtm_with_indices_a_dense_vs_sparse_1()

    performance_test_sizeA_100_sizeB_10000_num_connections_10000()
    #performance_test_sizeA_10000_sizeB_1000000_num_connections_1000000()
    #performance_test_sizeA_10000_sizeB_10000000_num_connections_1000000()
    #performance_test_sizeA_5000_sizeB_20000000_num_connections_1000000()
    #performance_test_sizeA_2000_sizeB_50000000_num_connections_1000000()
    #performance_test_sizeA_5000_sizeB_10000000_num_connections_2000000()
    #performance_test_sizeA_2000_sizeB_10000000_num_connections_5000000()
    #performance_test_sizeA_1000_sizeB_10000000_num_connections_10000000()
    #performance_test_sizeA_10000_sizeB_20000000_num_connections_1000000()
    performance_test_sizeA_20000_sizeB_10000000_num_connections_1000000()
    #performance_test_sizeA_40000_sizeB_10000000_num_connections_1000000()
    #performance_test_sizeA_10000_sizeB_10000000_num_connections_2000000()
    #performance_test_sizeA_10000_sizeB_10000000_num_connections_10000000()
    #performance_test_sizeA_80000_sizeB_10000000_num_connections_10000000()
    #performance_test_sizeA_10000_sizeB_10000000_num_connections_100000000()
    #performance_test_sizeA_10000_sizeB_20000000_num_connections_1000000_chunk_length_64_2_cutoff_0()


##                       PERFORMANCE RESULTS:                         ##
## -------------------------------------------------------------------##

# Takeaways:
# ----------
# Time is optimized at ~the number of virtual cores, but number of real cores is not bad
# Time: Mostly linear on A & C - (but sub-linear for both in certain circumstances)
#                                (no / nonlinear dependence on B)
# Space: Mostly linear on B & C (but sub-linear in certain circumstances)
#                               (essentially no dependence on A)

########################################################################
##         Processing time by number of processing cores              ##
########################################################################

# performance_test_sizeA_100_sizeB_10000_num_connections_10000:
#  1 core  ->   0.026948928833
#  2 cores ->   0.0255579948425
#  4 cores ->   0.0242738723755
#  6 cores ->   0.0301370620728
#  8 cores ->   0.0331609249115
# 12 cores ->   0.027214050293
# (These results seem to be mostly just noise)

# performance_test_sizeA_10000_sizeB_1000000_num_connections_1000000:
#  1 core  ->  51.6726438999
#  2 cores ->  31.5906500816
#  4 cores ->  22.2421519756
#  6 cores ->  22.1752150059
#  8 cores ->  20.3116240501  *** this is the number of virtual cores on my machine, seems optimal, but real cores could work too
# 12 cores ->  20.5109658241

########################################################################
##                     Collected Results Table                        ##
########################################################################

# sizeA     sizeB  num_connections   generation time   processing time  memory used (GB)
#   100     10000            10000   0.0256369113922   0.0361878871918  (very little)
# 10000   1000000          1000000   1.80626106262    20.1255269051     0.2
# 10000  10000000          1000000   6.75703692436    19.390857935      0.8
#  5000  20000000          1000000  12.0396170616     13.1896178722     1.5, peak 1.8
#  2000  50000000          1000000  30.3676009178      7.91083598137    3.5, peak 4.2
#  5000  10000000          2000000   8.96523094177    28.6623859406     0.9, peak 1.0
#  2000  10000000          5000000  14.0447731018     42.7930011749     1.1. peak 1.2
#  1000  10000000         10000000  22.9681668282     53.8616199493     1.5, peak 1.7
# 10000  20000000          1000000  12.3540771008     20.5963220596     1.5  
# 20000  10000000          1000000   7.13277387619    36.0603649616     0.8  
# 40000  10000000          1000000   6.98719000816    61.5007622242     0.8  
# 10000  10000000          2000000   9.17162895203    40.4926059246     0.9, peak 1.0
# 10000  10000000         10000000  22.7722630501    244.660306931      1.6, peak 1.7
# 80000  10000000         10000000  24.8473079205   1078.19644904       1.7
# 10000  10000000        100000000 175.944866896    2619.18887782      10.0, peak 11

# "chunk_size" doesn't seem affect the performance at these levels:
# 10000  20000000          1000000  12.3540771008     20.5963220596     1.5GB  chunk_size=1 
# 10000  20000000          1000000  12.6470808983     20.9645049572     1.5GB  chunk_size=2 

########################################################################
##                  Collected Results Breakdown                       ##
########################################################################
# sizeA, sizeB, num_connections, processing time, memory used (GB)

#--- Fix A, Fix B --------------------------------------------------------------
# Linear on C for time and space

# 10000, 10000000,   1000000,   19.390857935 ,  0.8
# 10000, 10000000,   2000000,   40.4926059246,  0.9
# 10000, 10000000,  10000000,  244.660306931 ,  1.6
# 10000, 10000000, 100000000, 2619.18887782  , 10.0


#--- Fix A, Fix C --------------------------------------------------------------
# Linear on B for space, no dependence for time (at least when B>C)

# 10000,  1000000, 1000000, 20.1255269051, 0.2
# 10000, 10000000, 1000000, 19.390857935 , 0.8
# 10000, 20000000, 1000000, 20.5963220596, 1.5


#--- Fix B, Fix C --------------------------------------------------------------
# Little dependence on A for space (for A << B and C)
# Sub-linear (log-ish?) dependence on A for time

# 10000, 10000000,     1000000,  19.390857935  , 0.8
# 20000, 10000000,     1000000,  36.0603649616 , 0.8
# 40000, 10000000,     1000000,  61.5007622242 , 0.8

#  1000, 10000000,    10000000,   53.8616199493, 1.5
# 10000, 10000000,    10000000,  244.660306931 , 1.6
# 80000, 10000000,    10000000, 1078.19644904  , 1.7

#  5000, 20000000,     1000000,   13.1896178722, 1.5
# 10000, 20000000,     1000000,   20.5963220596, 1.5

#  5000, 10000000,     2000000,   28.6623859406, 0.9
# 10000, 10000000,     2000000,   40.4926059246, 0.9


#--- Fix A, Flip B/C -----------------------------------------------------------
# Raising C by x when lowering B by x: increases time by ~x, decreases space by ~x

#  2000, 50000000, 1000000,  7.9108359814, 3.5
#  2000, 10000000, 5000000, 42.7930011749, 1.1

#  5000, 20000000, 1000000, 13.1896178722, 1.5
#  5000, 10000000, 2000000, 28.6623859406, 0.9

# 10000, 20000000, 1000000, 20.5963220596, 1.5
# 10000, 10000000, 2000000, 40.4926059246, 0.9


#--- Fix B, Flip A/C -----------------------------------------------------------
# Raising C by x when lowering A by x: increases time by ~1.5x, increases space by ~log(x)(???)

# 10000, 10000000,  1000000, 19.390857935 , 0.8
#  5000, 10000000,  2000000, 28.6623859406, 0.9
#  2000, 10000000,  5000000, 42.7930011749, 1.1
#  1000, 10000000, 10000000, 53.8616199493, 1.5

# 40000, 10000000,  1000000, 61.5007622242, 0.8                     
#  5000, 10000000,  2000000, 28.6623859406, 0.9


#--- Fix C, Flip A/B -----------------------------------------------------------
# Raising B by x when lowering A by x: decreases time by ~x, increases space by ~x

# 10000, 10000000, 1000000, 19.390857935 , 0.8
#  5000, 20000000, 1000000, 13.1896178722, 1.5
#  2000, 50000000, 1000000,  7.9108359814, 3.5


########################################################################
##             Performance comparison of recent branches              ##
########################################################################
# Tested this branch (v0.3.1) against 0.3.0fix and testing_alternate_union_count
# Looks like all have the same memory usage profile (0.85GB to 0.9GB, peak around 1.02GB)
# The timing of two newer branches is the same with the older version being slower:

# v0.3.1
#20000 10000000 1000000 9.99252605438 33.4487009048
#20000 10000000 1000000 10.0190310478 32.8482899666

# testing_alternate_union_count
#20000 10000000 1000000 10.1516039371 33.6840791702
#20000 10000000 1000000 10.1917278767 32.890652895

# v0.3.0fix
#20000 10000000 1000000 10.6243419647 36.6372568607
#20000 10000000 1000000 10.2242870331 36.3161420822
