import numpy as np
import matplotlib.pyplot as plt
import time
import os

init = time.process_time()

from data_generation import (independent_normal_uncalibrated, 
                             independent_normal_calibrated, 
                             independent_beta, 
                             dependent_normal_uncalibrated, 
                             dependent_normal_calibrated, 
                             dependent_beta)

selected = []
print()
print("Generating scores according to pre-set parameters. Select which distributions to generate from:")
print("_"*97)
print("|" + "|".join([s.center(15) for s in np.repeat(np.array(["Independent", "Dependent"]), 3, 0)]) + "|")
print("|" + "|".join([s.center(15) for s in np.tile(np.array(["Normal", "Normal", "Beta"]), 2)]) + "|")
print("|" + "|".join([s.center(15) for s in np.tile(np.array(["Uncalibrated", "Calibrated", " "]), 2)]) + "|")
# print("|".join([s.center(15) for s in (" ")*6]))
print("|" + "|".join(f"[{i+1}]".center(15) for i in range(6)) + "|")
print("\u203e"*97)
print("Type \"a\" to generate for all distributions")
print()

response = input()
print()

funs = np.array((independent_normal_uncalibrated,
        independent_normal_calibrated,
        independent_beta,
        dependent_normal_uncalibrated,
        dependent_normal_calibrated,
        dependent_beta))

if "a" in response:
    selected = funs
else:
    selected = funs[np.isin(list("123456"), list(response))]
    
if not os.path.exists("simulation_results"):
    os.makedirs("simulation_results")
    
for s in selected:
    print(f"Generating {s.__name__} scores...")
    start_time = time.process_time()
    
    N = 1000
    n = 10000
    k = n // 5
    r = 2
    
    Z_target, Z_decoys = s(N = N, 
                        n = n,
                        k = k,
                        decoys = r)

    (N, n) = Z_target[0].shape

    indicators = (Z_target[0] > Z_decoys[0])*1 - (Z_target[0] < Z_decoys[0])*1
    indicators = indicators + (indicators == 0) * np.random.choice((1, -1), indicators.shape)

    PT = [Z_target[0] * (indicators == 1) + Z_decoys[0] * (indicators == -1)]
    PT.append(indicators)
    PT.append(Z_target[1])

    indicators = (PT[0] > Z_decoys[1])*1 - (PT[0] < Z_decoys[1])*1
    indicators = indicators + (indicators == 0) * np.random.choice((1, -1), indicators.shape)

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

    res = 71
    alpha = np.repeat(np.linspace(0, 0.35, res), N*n).reshape(res, N, n)

    # print(FDR)
    # print(np.tile(FDR, (res, 1, 1)).reshape((res, N, n)))
    # print(alpha)

    # print((FDR <= alpha)*1)

    k = np.max(np.tile(np.arange(1, n+1), res*N).reshape(res, N, n) * (FDR <= alpha), axis=2).reshape(res, N, 1)

    # print(k)

    # raise(TypeError)

    # No longer works
    # plt.hist(np.min(FDR, axis=1), bins=40)

    # plt.show()

    selector = ((np.arange(1, n+1) <= k) & (W[1] == 1) & (W[2] == 1) * 1)

    FDP = np.sum((W[3]==0) & selector, axis=2).reshape((res, N, 1)) / np.clip(np.sum(selector, axis=2).reshape((res, N, 1)), 1, None)
    # print(FDP)

    discovery = np.sum((W[3] == 1) & selector, axis=2).reshape((res, N, 1)) / np.sum(W[3] == 1, axis=1).reshape((N, 1))
    # print(discovery)

    FDRbound = np.sum((W[1] == 1) & (W[2] == -1) & (np.arange(1, n+1) <= k), axis=2).reshape((res, N, 1)) / np.clip(np.sum((W[1] == 1) & (W[2] == 1) & (np.arange(1, n+1) <= k), axis=2).reshape((res, N, 1)), 1, None)

    if not os.path.exists(f"simulation_results/{s.__name__}"):
        os.makedirs(f"simulation_results/{s.__name__}")
        
    np.save(f"simulation_results/{s.__name__}/FDP.npy", FDP)
    np.save(f"simulation_results/{s.__name__}/discovery.npy", discovery)
    np.save(f"simulation_results/{s.__name__}/FDRbound.npy", FDRbound)

    # --- STANDARD TDC ---

    Wstd = [np.max((Z_target[0], Z_decoys[0], Z_decoys[1]), axis=0)]
    Wstd.append((np.argmax((Z_target[0], Z_decoys[0], Z_decoys[1]), axis=0) == 0)*2 - 1)
    Wstd.append(Z_target[1])

    idx = np.argsort(Wstd[0], axis=1, descending=True)

    Wstd[0] = np.take_along_axis(Wstd[0], idx, axis=1)
    Wstd[1] = np.take_along_axis(Wstd[1], idx, axis=1)
    Wstd[2] = np.take_along_axis(Wstd[2], idx, axis=1)

    T = np.cumsum(Wstd[1] == 1, axis=1)
    D = np.cumsum(Wstd[1] == -1, axis=1)

    FDRstd = (D + 1)*0.5/np.clip(T, 1, None)

    kstd = np.max(np.tile(np.arange(1, n+1), res*N).reshape(res, N, n) * (FDRstd <= alpha), axis=2).reshape(res, N, 1)

    selector = ((np.arange(1, n+1) <= kstd) & (Wstd[1] == 1) * 1)

    # print(np.sum((Wstd[2] == 0) & selector, axis=2).reshape((res, N, 1)))
    # print(np.clip(np.sum(selector, axis=2).reshape((res, N, 1)), 1, None))

    FDPstd = np.sum((Wstd[2] == 0) & selector, axis=2).reshape((res, N, 1)) / np.clip(np.sum(selector, axis=2).reshape((res, N, 1)), 1, None)

    # print(FDPstd)

    discoverystd = np.sum((Wstd[2] == 1) & selector, axis=2).reshape((res, N, 1)) / np.sum(Wstd[2] == 1, axis=1).reshape((N, 1))

    np.save(f"simulation_results/{s.__name__}/FDPstd.npy", FDPstd)
    np.save(f"simulation_results/{s.__name__}/discoverystd.npy", discoverystd)
    
    print(f"Done in {(time.process_time() - start_time):.3f} seconds.")
    
if len(selected) > 1:
    print(f"All done! Total time {(time.process_time() - init):.3f} seconds.")