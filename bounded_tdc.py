import numpy as np
import matplotlib.pyplot as plt

from independent_normal import gen_scores

N = 3
n = 10
k = n // 5
r = 2

Z_target, Z_decoys = gen_scores(N = N, 
                    n = n,
                    k = k,
                    decoys = r)

# alpha = 0.3

indicators = (Z_target[0] > Z_decoys[0])*1 - (Z_target[0] < Z_decoys[0])*1

PT = [Z_target[0] * (indicators == 1) + Z_decoys[0] * (indicators == -1)]
PT.append(indicators)
PT.append(Z_target[1])

indicators = (PT[0] > Z_decoys[1])*1 - (PT[0] < Z_decoys[1])*1

W = [PT[0] * (indicators == 1) + Z_decoys[1] * (indicators == -1)]

idx = np.argsort(W[0], axis=1, descending=True)

W[0] = np.take_along_axis(W[0], idx, axis=1)

W.append(np.take_along_axis(indicators, idx, axis=1))
W.append(np.take_along_axis(PT[1], idx, axis=1))
W.append(np.take_along_axis(PT[2], idx, axis=1))

D2 = np.cumsum(W[1] == -1, axis=1)
T2 = np.cumsum(W[1] == 1, axis=1)
D1 = np.cumsum((W[1] == 1) & (W[2] == -1), axis=1)

FDR = (D2 + 0.5) / np.clip(T2-D2, 1, None)

# # Graph of Minimum FDR
# plt.hist(np.min(FDR, axis=1), bins=40)

# plt.show()

res = 2
alpha = np.repeat(np.linspace(0.2, 0.4, res), N*n).reshape(res, N, n)

print(FDR)
print(np.tile(FDR, (res, 1, 1)).reshape((res, N, n)))
print(alpha)

print((FDR <= alpha)*1)

k = np.max(np.tile(np.arange(1, n+1), res*N).reshape(res, N, n) * (FDR <= alpha), axis=2).reshape(res, N, 1)

print(k)

# raise(TypeError)

# # No longer works
# plt.hist(np.min(FDR, axis=1), bins=40)

# plt.show()

selector = ((np.arange(1, n+1) <= k) & (W[1] == 1) & (W[2] == 1) * 1)

FDP = np.sum((W[3]==0) & selector, axis=2).reshape((res, N, 1)) / np.max((np.ones((res, N, 1)), np.sum(selector, axis=2).reshape((res, N, 1))), axis=0)
print(FDP)

discovery = np.sum((W[3] == 1) & selector, axis=2).reshape((res, N, 1)) / np.sum(W[3] == 1, axis=1).reshape((N, 1))
print(discovery)

FDRbound = np.sum((W[1] == 1) & (W[2] == -1), axis=1).reshape((N, 1)) / np.sum((W[1] == 1) & (W[2] == 1), axis=1).reshape((N, 1))
print(FDRbound)

