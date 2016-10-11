'''Test it all'''

import mtm_stats
import numpy as np

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

def test_get_Jaccard_index_1():
    bcd, ji = mtm_stats.get_Jaccard_index(TEST_SET_1)
    assert bcd == {'a1': 3, 'a3': 1, 'a2': 2, 'a4': 1}
    assert ji == {('a1', 'a2'): 2./3, ('a1', 'a3'): 1./3}

def timing_test_1():
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=100,
                                                     sizeB=10000,
                                                     num_connections=10000,)
    assert mt < 1

def timing_test_2():
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=10000,
                                                     sizeB=1000000,
                                                     num_connections=1000000,)
    assert mt < 100, 'mtm_stats is too slow'
    # Uses about 0.2GB, 20s (on my machine)


def timing_test_3():
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=10000,
                                                     sizeB=10000000,
                                                     num_connections=1000000,)
    assert mt < 100, 'mtm_stats is too slow'
    # Uses about 0.8GB, 19s (on my machine)

def timing_test_4():
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=5000,
                                                     sizeB=20000000,
                                                     num_connections=1000000,)
    assert mt < 100, 'mtm_stats is too slow'
    # Uses about 1.5GB, 13s (on my machine)

def timing_test_5():
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=2000,
                                                     sizeB=50000000,
                                                     num_connections=1000000,)
    assert mt < 100, 'mtm_stats is too slow'
    # Uses about 3.5GB, 8s (on my machine)

def timing_test_6():
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=5000,
                                                     sizeB=10000000,
                                                     num_connections=2000000,)
    assert mt < 100, 'mtm_stats is too slow'
    # Uses about 0.9GB, 29s (on my machine)

def timing_test_7():
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=2000,
                                                     sizeB=10000000,
                                                     num_connections=5000000,)
    assert mt < 100, 'mtm_stats is too slow'
    # Uses about 1.1GB, 43s (on my machine)

def timing_test_8():
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=1000,
                                                     sizeB=10000000,
                                                     num_connections=10000000,)
    assert mt < 100, 'mtm_stats is too slow'
    # Uses about 15GB, 54s (on my machine)

def stress_test_1():
    '''Pushing it until it breaks :) '''
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=10000,
                                                     sizeB=20000000,
                                                     num_connections=1000000,
                                                     chunk_length_64=1,
                                                     cutoff=0)
    # Uses about 1.5GB, 21s (on my machine)

def stress_test_2():
    '''Pushing it until it breaks :) '''
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=10000,
                                                     sizeB=20000000,
                                                     num_connections=1000000,
                                                     chunk_length_64=2,
                                                     cutoff=0)
    # Uses about 1.5GB, 21s (on my machine)

def stress_test_3():
    '''Pushing it until it breaks :) '''
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=20000,
                                                     sizeB=10000000,
                                                     num_connections=1000000,
                                                     chunk_length_64=1,
                                                     cutoff=0)
    # Uses about 0.8GB, 36s (on my machine)

def stress_test_4():
    '''Pushing it until it breaks :) '''
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=40000,
                                                     sizeB=10000000,
                                                     num_connections=1000000,
                                                     chunk_length_64=1,
                                                     cutoff=0)
    # Uses about 0.8GB, 62s (on my machine)

def stress_test_5():
    '''Pushing it until it breaks :) '''
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=10000,
                                                     sizeB=10000000,
                                                     num_connections=2000000,
                                                     chunk_length_64=1,
                                                     cutoff=0)
    # Uses about 0.9GB, 40s (on my machine)

def stress_test_6():
    '''Pushing it until it breaks :) '''
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=10000,
                                                     sizeB=10000000,
                                                     num_connections=10000000,
                                                     chunk_length_64=1,
                                                     cutoff=0)
    # Uses about 1.6GB, 245s (on my machine)

def stress_test_7():
    '''Pushing it until it breaks :) '''
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=80000,
                                                     sizeB=10000000,
                                                     num_connections=10000000,
                                                     chunk_length_64=1,
                                                     cutoff=0)
    # Yeah, so it broke -- not enough memory to even make the binary input!
    # This issue actually could be worked around at the cost of some setup
    # speed by looping through the array multiple times to generate subsets
    # and compressing them in smaller pieces
    # Update: the new sba conversion function doesn't crash
    # Uses about 1.7GB, 1078s (on my machine)

def stress_test_8():
    '''Pushing it until it breaks :) '''
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=10000,
                                                     sizeB=10000000,
                                                     num_connections=100000000,
                                                     chunk_length_64=1,
                                                     cutoff=0)
    # Uses about 10.0GB, 2619s (on my machine)

if __name__ == '__main__':
    test_mtm_stats_1()
    test_get_Jaccard_index_1()
    timing_test_1()
    #timing_test_8()
    #stress_test_8()



##                       TIMING RESULTS:                              ##
## -------------------------------------------------------------------##


# Takeaways:
# Time is optimized at ~the number of virtual cores, but number of real cores is not bad
# Time: Linear on A & C - (but sub-linear for C in certain circumstances)
#                         (no / nonlinear dependence on B)
# Space: Mostly linear on B & C (but sub-linear in certain circumstances)
#                               (essentially no dependence on A)

########################################################################
##                   By number of processing cores                    ##
########################################################################

# 1 core:
# timing_test_1 -> 0.0268700122833 0.026948928833
# timing_test_2 -> 7.23184394836 51.6726438999
# 2 cores:
# timing_test_1 -> 0.0255460739136 0.0255579948425
# timing_test_2 -> 31.5906500816
# 4 cores:
# timing_test_1 -> 0.0256078243256 0.0242738723755
# timing_test_2 -> 7.42782711983 22.2421519756
# 6 cores:
# timing_test_1 -> 0.0282258987427 0.0301370620728
# timing_test_2 -> 7.36188602448 22.1752150059
# 8 cores:                                          *** this is the number of virtual cores on my machine, seems optimal, but real cores could work too
# timing_test_1 -> 0.025887966156 0.0331609249115
# timing_test_2 -> 7.10551309586 20.3116240501
# 12 cores:
# timing_test_1 -> 0.0281281471252 0.027214050293
# timing_test_2 -> 7.5020840168 20.5109658241

########################################################################
##                     Collected Results Table                        ##
########################################################################

# timing_test_1 ->   100    10000     10000   0.0256369113922 0.0361878871918 -- very little memory
# timing_test_2 -> 10000  1000000   1000000   1.80626106262  20.1255269051  -- 0.2GB
# timing_test_3 -> 10000 10000000   1000000   6.75703692436  19.390857935   -- 0.8GB
# timing_test_4 ->  5000 20000000   1000000  12.0396170616   13.1896178722  -- 1.5GB, peak 1.8
# timing_test_5 ->  2000 50000000   1000000  30.3676009178    7.91083598137 -- 3.5GB, peak 4.2
# timing_test_6 ->  5000 10000000   2000000   8.96523094177  28.6623859406  -- 0.9GB, peak 1.0
# timing_test_7 ->  2000 10000000   5000000  14.0447731018   42.7930011749  -- 1.1GB. peak 1.2
# timing_test_8 ->  1000 10000000  10000000  22.9681668282   53.8616199493  -- 1.5GB, peak 1.7
# stress_test_1 -> 10000 20000000   1000000  12.3540771008   20.5963220596  -- 1.5GB
# stress_test_2 -> 10000 20000000   1000000  12.6470808983   20.9645049572  -- 1.5GB, peak 1.8
# stress_test_3 -> 20000 10000000   1000000   7.13277387619  36.0603649616  -- 0.8GB
# stress_test_4 -> 40000 10000000   1000000   6.98719000816  61.5007622242  -- 0.8GB
# stress_test_5 -> 10000 10000000   2000000   9.17162895203  40.4926059246  -- 0.9GB, peak 1.0
# stress_test_6 -> 10000 10000000  10000000  22.7722630501  244.660306931   -- 1.6GB, peak 1.7
# stress_test_7 -> 80000 10000000  10000000  24.8473079205 1078.19644904    -- 1.7GB
# stress_test_8 -> 10000 10000000 100000000 175.944866896  2619.18887782    -- 10.0GB, peak 11

########################################################################
##                  Collected Results Breakdown                       ##
########################################################################

#--- Fix A, Fix B --------------------------------------------------------------
# Linear on C for time and space

#timing_test_3  10000   10000000    1000000     19.390857935    --  0.8GB
#stress_test_5  10000   10000000    2000000     40.4926059246   --  0.9GB
#stress_test_6  10000   10000000    10000000    244.660306931   --  1.6GB
#stress_test_8  10000   10000000    100000000   2619.18887782   --  10.0GB


#--- Fix A, Fix C --------------------------------------------------------------
# Linear on B for space, no dependence for time (at least when B>C)

#timing_test_2  10000   1000000     1000000     20.1255269051   --  0.2GB
#timing_test_3  10000   10000000    1000000     19.390857935    --  0.8GB
#stress_test_1  10000   20000000    1000000     20.5963220596   --  1.5GB


#--- Fix B, Fix C --------------------------------------------------------------
# Little dependence on A for space (for A << B and C)
# Sub-linear (log-ish?) dependence on A for time

#timing_test_3  10000   10000000    1000000     19.390857935    --  0.8GB
#stress_test_3  20000   10000000    1000000     36.0603649616   --  0.8GB
#stress_test_4  40000   10000000    1000000     61.5007622242   --  0.8GB

#timing_test_8  1000    10000000    10000000    53.8616199493   --  1.5GB
#stress_test_6  10000   10000000    10000000    244.660306931   --  1.6GB
#stress_test_7  80000   10000000    10000000    1078.19644904   --  1.7GB

#timing_test_4  5000    20000000    1000000     13.1896178722   --  1.5GB
#stress_test_1  10000   20000000    1000000     20.5963220596   --  1.5GB

#timing_test_6  5000    10000000    2000000     28.6623859406   --  0.9GB
#stress_test_5  10000   10000000    2000000     40.4926059246   --  0.9GB


#--- Fix A, Flip B/C -----------------------------------------------------------
# Raising C by x when lowering B by x: increases time by ~x, decreases space by ~x

#timing_test_5  2000    50000000    1000000     7.9108359814    --  3.5GB
#timing_test_7  2000    10000000    5000000     42.7930011749   --  1.1GB

#timing_test_4  5000    20000000    1000000     13.1896178722   --  1.5GB
#timing_test_6  5000    10000000    2000000     28.6623859406   --  0.9GB

#stress_test_1  10000   20000000    1000000     20.5963220596   --  1.5GB
#stress_test_5  10000   10000000    2000000     40.4926059246   --  0.9GB


#--- Fix B, Flip A/C -----------------------------------------------------------
# Raising C by x when lowering A by x: increases time by ~1.5x, increases space by ~log(x)(???)

#timing_test_3  10000   10000000    1000000     19.390857935    --  0.8GB
#timing_test_6  5000    10000000    2000000     28.6623859406   --  0.9GB
#timing_test_7  2000    10000000    5000000     42.7930011749   --  1.1GB
#timing_test_8  1000    10000000    10000000    53.8616199493   --  1.5GB

#stress_test_4  40000   10000000    1000000     61.5007622242   --  0.8GB                     
#timing_test_6  5000    10000000    2000000     28.6623859406   --  0.9GB


#--- Fix C, Flip A/B -----------------------------------------------------------
# Raising B by x when lowering A by x: decreases time by ~x, increases space by ~x

#timing_test_3  10000   10000000    1000000     19.390857935    --  0.8GB
#timing_test_4  5000    20000000    1000000     13.1896178722   --  1.5GB
#timing_test_5  2000    50000000    1000000     7.9108359814    --  3.5GB
