import numpy as np
import matplotlib.pyplot as plt

alpha = np.linspace(0, 0.35, 71)

FDP = np.load("simulation_results/FDP.npy")

discovery = np.load("simulation_results/discovery.npy")

FDRbound = np.load("simulation_results/FDRbound.npy")

FDPstd = np.load("simulation_results/FDPstd.npy")

discoverystd = np.load("simulation_results/discoverystd.npy")

print(FDP.shape, discovery.shape, FDRbound.shape)

fig1data = FDRbound - FDP

fig, axs = plt.subplots(1, 4, figsize=(15, 3.5))

# # First plot
axs[0].plot(alpha, np.mean(fig1data, axis=1))
axs[0].plot(alpha, np.mean(fig1data, axis=1) + np.std(fig1data, axis=1))
axs[0].plot(alpha, np.mean(fig1data, axis=1) - np.std(fig1data, axis=1))
axs[0].grid()

# # Second plot
axs[1].plot(alpha, np.mean(FDP, axis=1), color="red")
axs[1].plot(alpha, np.mean(FDP, axis=1) + np.std(FDP, axis=1), color="red", linestyle="--")
axs[1].plot(alpha, np.mean(FDP, axis=1) - np.std(FDP, axis=1), color="red", linestyle="--")
axs[1].plot(alpha, np.mean(FDPstd, axis=1), color="blue")
axs[1].plot(alpha, np.mean(FDPstd, axis=1) + np.std(FDPstd, axis=1), color="blue", linestyle="--")
axs[1].plot(alpha, np.mean(FDPstd, axis=1) - np.std(FDPstd, axis=1), color="blue", linestyle="--")
axs[1].plot(alpha, alpha, color="green", linewidth=3)
axs[1].grid()
# plt.hist(FDPstd[14], bins=40)

# Third Plot
axs[2].plot(alpha, np.mean(discovery, axis=1))
axs[2].plot(alpha, np.mean(discoverystd, axis = 1))
axs[2].grid()

axs[3].plot(alpha, np.mean(discovery, axis=1)/np.mean(discoverystd, axis=1))
axs[3].set_ylim([0.95, 1.05])
# plt.semilogy()
axs[3].grid()

plt.show()