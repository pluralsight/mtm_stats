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
    # This uses about 0.5GB on my machine
    # This also takes about 19s


def timing_test_3():
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=10000,
                                                     sizeB=10000000,
                                                     num_connections=1000000,)
    assert mt < 100, 'mtm_stats is too slow'
    # This uses about 1.5GB on my machine
    # This also takes about 19s

def timing_test_4():
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=5000,
                                                     sizeB=20000000,
                                                     num_connections=1000000,)
    assert mt < 100, 'mtm_stats is too slow'
    # This uses about 3.5GB on my machine (4.5GB peak during setup)
    # This also takes about 8s

def timing_test_5():
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=2000,
                                                     sizeB=50000000,
                                                     num_connections=1000000,)
    assert mt < 100, 'mtm_stats is too slow'
    # This uses about 2GB on my machine
    # This also takes about 12s

def timing_test_6():
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=5000,
                                                     sizeB=10000000,
                                                     num_connections=2000000,)
    assert mt < 100, 'mtm_stats is too slow'
    # This uses about 1.6GB on my machine
    # This also takes about 26s

def timing_test_7():
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=2000,
                                                     sizeB=10000000,
                                                     num_connections=5000000,)
    assert mt < 100, 'mtm_stats is too slow'
    # This uses about 1.5GB on my machine
    # This also takes about 41s

def timing_test_8():
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=1000,
                                                     sizeB=10000000,
                                                     num_connections=10000000,)
    assert mt < 100, 'mtm_stats is too slow'
    # This uses about 1.8GB on my machine
    # This also takes about 56s

def stress_test_1():
    '''Pushing it until it breaks :) '''
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=10000,
                                                     sizeB=20000000,
                                                     num_connections=1000000,
                                                     chunk_length_64=1,
                                                     cutoff=0)
    # This uses about 2.3GB on my machine
    # Also uses only 19s

def stress_test_2():
    '''Pushing it until it breaks :) '''
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=10000,
                                                     sizeB=20000000,
                                                     num_connections=1000000,
                                                     chunk_length_64=2,
                                                     cutoff=0)
    # This uses about 2.3GB on my machine
    # Also uses only 19s

def stress_test_3():
    '''Pushing it until it breaks :) '''
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=20000,
                                                     sizeB=10000000,
                                                     num_connections=1000000,
                                                     chunk_length_64=1,
                                                     cutoff=0)
    # This uses about 2.2GB on my machine
    # This takes about 33s

def stress_test_4():
    '''Pushing it until it breaks :) '''
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=40000,
                                                     sizeB=10000000,
                                                     num_connections=1000000,
                                                     chunk_length_64=1,
                                                     cutoff=0)
    # This uses about 3.4GB on my machine
    # This takes about 64s

def stress_test_5():
    '''Pushing it until it breaks :) '''
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=10000,
                                                     sizeB=10000000,
                                                     num_connections=2000000,
                                                     chunk_length_64=1,
                                                     cutoff=0)
    # This uses about 2.2GB on my machine
    # This takes about 40s

def stress_test_6():
    '''Pushing it until it breaks :) '''
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=10000,
                                                     sizeB=10000000,
                                                     num_connections=10000000,
                                                     chunk_length_64=1,
                                                     cutoff=0)
    # This uses about 5GB on my machine
    # This one also takes a whopping 254s

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

def stress_test_8():
    '''Pushing it until it breaks :) '''
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=10000,
                                                     sizeB=10000000,
                                                     num_connections=200000000,
                                                     chunk_length_64=1,
                                                     cutoff=0)

# Takeaways:
# Space scales linearly with the size of A
# The initial space scales linearly with the size of B, but not the final size
# Space scales logartithmically (maybe) with the size of B
# Space scales non-linearly with the number of connections (depends on the relative size of B & C)
# Time scales linearly with the number of connections
# The size of B has little to do with the time

# Time: Linear A & C (no dependence / weird dependence on B)
# Space: Linear A, (non-linear on B & C)


if __name__ == '__main__':
    test_mtm_stats_1()
    test_get_Jaccard_index_1()
    timing_test_1()
    timing_test_8()
    #stress_test_4()
    
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
    # 8 cores:                                          *** this is the number of virtual cores on my machine, seems optimal
    # timing_test_1 -> 0.025887966156 0.0331609249115
    # timing_test_2 -> 7.10551309586 20.3116240501
    # 12 cores:
    # timing_test_1 -> 0.0281281471252 0.027214050293
    # timing_test_2 -> 7.5020840168 20.5109658241
