import numpy as np
import matplotlib.pyplot as plt

alpha = np.linspace(0, 0.35, 100)

FDP = np.load("simulation_results/FDP.npy")

discovery = np.load("simulation_results/discovery.npy")

FDRbound = np.load("simulation_results/FDRbound.npy")

FDPstd = np.load("simulation_results/FDPstd.npy")

discoverystd = np.load("simulation_results/discoverystd.npy")

print(FDP.shape, discovery.shape, FDRbound.shape)

fig1data = FDRbound - FDP

# # First plot
# plt.plot(alpha, np.mean(fig1data, axis=1))
# plt.plot(alpha, np.mean(fig1data, axis=1) + np.std(fig1data, axis=1))
# plt.plot(alpha, np.mean(fig1data, axis=1) - np.std(fig1data, axis=1))

# # Second plot
# plt.plot(alpha, np.mean(FDPstd, axis=1))
# plt.plot(alpha, np.mean(FDPstd, axis=1) + np.std(FDPstd, axis=1))
# plt.plot(alpha, np.mean(FDPstd, axis=1) - np.std(FDPstd, axis=1))
# plt.plot(alpha, alpha)

# Third Plot
# plt.plot(alpha, np.mean(discovery, axis=1))
# plt.plot(alpha, np.mean(discoverystd, axis = 1))

# plt.plot(alpha, np.mean(discovery, axis=1)/np.mean(discoverystd, axis=1))
# plt.ylim(0.5, 1.50)
# plt.semilogy()

plt.show()