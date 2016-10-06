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
                                                     sizeB = 10000,
                                                     num_connections = 10000,)
    assert mt < 1

def timing_test_2():
    gt, mt = mtm_stats.testing_utils.run_timing_test(sizeA=10000,
                                                     sizeB = 10000000,
                                                     num_connections = 1000000,)
    assert mt < 100, 'mtm_stats is too slow'

if __name__ == '__main__':
    test_mtm_stats_1()
    test_get_Jaccard_index_1()
    timing_test_1()
    timing_test_2()
