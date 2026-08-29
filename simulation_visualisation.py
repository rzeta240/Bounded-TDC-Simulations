import numpy as np
import matplotlib.pyplot as plt

alpha = np.linspace(0, 0.35, 71)

print()
print("Visualising all available metrics for specified distribution. Please select a distribution:")
print("_"*97)
print("|" + "|".join([s.center(15) for s in np.repeat(np.array(["Independent", "Dependent"]), 3, 0)]) + "|")
print("|" + "|".join([s.center(15) for s in np.tile(np.array(["Normal", "Normal", "Beta"]), 2)]) + "|")
print("|" + "|".join([s.center(15) for s in np.tile(np.array(["Uncalibrated", "Calibrated", " "]), 2)]) + "|")
# print("|".join([s.center(15) for s in (" ")*6]))
print("|" + "|".join(f"[{i+1}]".center(15) for i in range(6)) + "|")
print("\u203e"*97)

sims = ("independent_normal_uncalibrated", 
        "independent_normal_calibrated", 
        "independent_beta", 
        "dependent_normal_uncalibrated", 
        "dependent_normal_calibrated", 
        "dependent_beta")

select = input()

f = sims[int(select)-1]

FDP = np.load(f"simulation_results/{f}/FDP.npy")

discovery = np.load(f"simulation_results/{f}/discovery.npy")

FDRbound = np.load(f"simulation_results/{f}/FDRbound.npy")

FDPstd = np.load(f"simulation_results/{f}/FDPstd.npy")

discoverystd = np.load(f"simulation_results/{f}/discoverystd.npy")

print(FDP.shape, discovery.shape, FDRbound.shape)

fig1data = FDRbound - FDP

fig, axs = plt.subplots(1, 4, figsize=(15, 3.5))

# # First plot
axs[0].plot(alpha, np.mean(fig1data, axis=1))
axs[0].plot(alpha, np.quantile(fig1data, 0.75, axis=1))
axs[0].plot(alpha, np.quantile(fig1data, 0.25, axis=1))
axs[0].grid()
axs[0].legend(["Mean", "Q3", "Q1"])
axs[0].set_title("Mean(FDR_bound - FDP)")

# # Second plot
line1, = axs[1].plot(alpha, np.quantile(FDP, 0.5, axis=1), color="red")
axs[1].plot(alpha, np.quantile(FDP, 0.75, axis=1), color="red", linestyle="--")
axs[1].plot(alpha, np.quantile(FDP, 0.25, axis=1), color="red", linestyle="--")
line2, = axs[1].plot(alpha, np.quantile(FDPstd, 0.5, axis=1), color="blue")
axs[1].plot(alpha, np.quantile(FDPstd, 0.75, axis=1), color="blue", linestyle="--")
axs[1].plot(alpha, np.quantile(FDPstd, 0.25, axis=1), color="blue", linestyle="--")
line3, = axs[1].plot(alpha, alpha, color="green", linewidth=3)
axs[1].grid()
axs[1].legend([line1, line2, line3], ["Bounded TDC", "Standard TDC", "y=x"])
axs[1].set_title("Mean(FDP)")
# plt.hist(FDPstd[14], bins=40)

# Third Plot
line1, = axs[2].plot(alpha, np.mean(discovery, axis=1))
line2, = axs[2].plot(alpha, np.mean(discoverystd, axis = 1))
axs[2].legend([line1, line2], ["Bounded TDC", "Standard TDC"])
axs[2].set_title("Discovery power")
axs[2].grid()

axs[3].plot(alpha, np.mean(discovery, axis=1)/np.mean(discoverystd, axis=1))
axs[3].set_ylim([0.95, 1.05])
axs[3].set_title("Ratio of discovery powers\n(Bounded/Standard)")
# plt.semilogy()
axs[3].grid()

plt.suptitle(f)

plt.tight_layout()

plt.show()