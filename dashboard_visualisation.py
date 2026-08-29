import numpy as np
import matplotlib.pyplot as plt

alpha = np.linspace(0, 0.35, 71)

sims = ("independent_normal_uncalibrated", 
        "independent_normal_calibrated", 
        "independent_beta", 
        "dependent_normal_uncalibrated", 
        "dependent_normal_calibrated", 
        "dependent_beta")

FDPs = []
discoveries = []
FDRbounds = []
FDPsstd = []
discoveriesstd = []

for s in sims:
    FDPs.append(np.load(f"simulation_results/{s}/FDP.npy"))
    discoveries.append(np.load(f"simulation_results/{s}/discovery.npy"))
    FDRbounds.append(np.load(f"simulation_results/{s}/FDRbound.npy"))
    FDPsstd.append(np.load(f"simulation_results/{s}/FDPstd.npy"))
    discoveriesstd.append(np.load(f"simulation_results/{s}/discoverystd.npy"))

fig, axs = plt.subplots(4, 6, figsize=(14.5, 7))

axs = np.array(axs).reshape((4, 6))
fig.tight_layout()

for i, s in enumerate(sims):
    fig1data = FDRbounds[i] - FDPs[i]

    # # First plot
    axs[0, i].plot(alpha, np.mean(fig1data, axis=1))
    axs[0, i].plot(alpha, np.mean(fig1data, axis=1) + np.std(fig1data, axis=1))
    axs[0, i].plot(alpha, np.mean(fig1data, axis=1) - np.std(fig1data, axis=1))
    axs[0, i].grid()

    # # Second plot
    axs[1, i].plot(alpha, np.mean(FDPs[i], axis=1), color="red")
    axs[1, i].plot(alpha, np.mean(FDPs[i], axis=1) + np.std(FDPs[i], axis=1), color="red", linestyle="--")
    axs[1, i].plot(alpha, np.mean(FDPs[i], axis=1) - np.std(FDPs[i], axis=1), color="red", linestyle="--")
    axs[1, i].plot(alpha, np.mean(FDPsstd[i], axis=1), color="blue")
    axs[1, i].plot(alpha, np.mean(FDPsstd[i], axis=1) + np.std(FDPsstd[i], axis=1), color="blue", linestyle="--")
    axs[1, i].plot(alpha, np.mean(FDPsstd[i], axis=1) - np.std(FDPsstd[i], axis=1), color="blue", linestyle="--")
    axs[1, i].plot(alpha, alpha, color="green", linewidth=3)
    axs[1, i].grid()
    # plt.hist(FDPsstd[i][14], bins=40)

    # Third Plot
    axs[2, i].plot(alpha, np.mean(discoveries[i], axis=1))
    axs[2, i].plot(alpha, np.mean(discoveriesstd[i], axis = 1))
    axs[2, i].grid()

    axs[3, i].plot(alpha, np.mean(discoveries[i], axis=1)/np.mean(discoveriesstd[i], axis=1))
    axs[3, i].set_ylim([0.95, 1.05])
    # plt.semilogy()
    axs[3, i].grid()

fig.tight_layout()
plt.show()