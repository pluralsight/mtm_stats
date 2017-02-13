from __future__ import print_function
from mtm_stats import mtm_stats

bc, iuc = mtm_stats([('a1', 'b1'),
                     ('a1', 'b2'),
                     ('a1', 'b3'),
                     ('a2', 'b1'),
                     ('a2', 'b2'),
                     ('a3', 'b3'),
                     ('a4', 'b9'),])
print(bc)
print(iuc)
