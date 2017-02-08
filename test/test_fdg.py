import mtm_stats
import numpy as np
import render_d3_fdg
from future.utils import viewitems

connections = mtm_stats.testing_utils.generate_test_set(sizeA=1000,
                                                        sizeB=100000,
                                                        num_connections=10000)
bc, j = mtm_stats.get_Jaccard_index(connections)
flatten = lambda x: [j for i in x for j in i]
nodes = [(i, 1) for i in sorted(set(flatten(j.keys())))]
links = [k + (v,) for k, v in viewitems(j)]
render_d3_fdg.fdg(nodes, links, slider_init_x=0)
