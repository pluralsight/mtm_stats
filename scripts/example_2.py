from __future__ import print_function
from mtm_stats import mtm_stats_iterator

bc_g, iuc_gg = mtm_stats_iterator([('a1', 'b1'),
                            ('a1', 'b2'),
                            ('a1', 'b3'),
                            ('a2', 'b1'),
                            ('a2', 'b2'),
                            ('a3', 'b3'),
                            ('a4', 'b9'),],
                           2)
bc = dict(bc_g)
iuc = {(i,j): (ic,uc)
       for iuc_g in iuc_gg
       for i,j,ic,uc in iuc_g} 

print(bc)
print(iuc)
