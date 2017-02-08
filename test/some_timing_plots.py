from builtins import zip
#!/usr/bin/env python
import matplotlib.pyplot as plt

# Fix A, Fix B -- Linear on C for time,
#                 Linear with offset on C for space:
C, s, t = zip([1000000, 19.390857935, 0.8],
              [2000000, 40.4926059246, 0.9],
              [10000000, 244.660306931, 1.6],
              [100000000, 2619.18887782, 10.0])
plt.figure(1)
plt.subplot(121)
plt.plot(C, s, '.-')
plt.subplot(122)
plt.plot(C, t, '.-')

# Fix A, Fix C -- *No dependence* on B for time,
#                 Linear with offset on B for space
B, s, t = zip([1000000, 20.1255269051, 0.2],
              [10000000, 19.390857935, 0.8],
              [20000000, 20.5963220596, 1.5])
plt.figure(2)
plt.subplot(121)
plt.plot(B, s, '.-')
plt.subplot(122)
plt.plot(B, t, '.-')

# Fix B, Fix C -- Sub-linear with offset on A for time,
#                 *No dependence* on A for space
plt.figure(3)
plt.plot([1000, 10000, 80000],
         [53.8616199493, 244.660306931, 1078.19644904],
         '.-')

plt.show()
