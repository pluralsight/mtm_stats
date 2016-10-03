mtm_stats
Highly efficient set statistics about many-to-many relationships


This module computes statistics (counts) in the common case where two sets (A and B) have a many-to-many relationship. Specifically, it computes:

* *connection count* (or just *count*): the number of connections a member of A has to B
* *intersection count*: the number of connections to B that two members of A share in common
* *union count*: the number of connections to B that two members of A have in total

This module aims to compute these for every possible combination of members in A as fast as possible and without approximation.

If N_A is the length of A, the connection count will have size N_A and the others two will each have size N_A * (N_A - 1) / 2.

Any other set property can be derived from these. The single-sided difference will be:
difference(A1, A2) = count(A1) - intersection(A1, A2)
And the symmetric difference is just (union-intersection).

The input for this module is a list of A/B pairs, i.e.:
[(a1, b1), (a1, b2), (a2, b2), ...]

This will be the most effective when the connections between A and B are sparse.

In addition, there is one optional approximation, a cutoff filter for small intersections (default is 0). If cutoff > 0, any result with intersection count <= cutoff will be approximated as 0; the union count is thhen assumed to be count(A1) + count(A2).

Implementation details:
This module uses a combination of techniques to achieve this goal:
* Core implemented in C and Cython
* Bit packing (members of B are each assigned one bit)
* A sparse block array compression scheme that ignores large blocks of zeros in the bit-packed sets
* Efficiently applied bit operations (&, |, popcount)
* Only storing intersections and unions for A/A pairs that have nonzero intersection [(A1, A2, intersection_count, union_count)...]
* The optional cutoff describes above

These appproximations not only give huge time savings over a brute-force implementation, but also cut down on memory usage (which could easily crash a machine with a sets as small as 100K elements).

💥

This technique will work well until the size of N^2 becomes unmanagely large. In a case like that, you should try a probabalistic algorithm like Count Min Sketch (see:
for a faster but less accurate implementation, check out Count Min Sketch:
https://redislabs.com/blog/count-min-sketch-the-art-and-science-of-estimating-stuff#.V-6Mctz3aaM
and
https://tech.shareaholic.com/2012/12/03/the-count-min-sketch-how-to-count-over-large-keyspaces-when-about-right-is-good-enough/
)
