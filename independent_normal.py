import numpy as np
import matplotlib.pyplot as plt

def gen_scores(N = 1000, n = 10000, k = 2000, decoys = 2, hyper_v = 0.35, hyper_mu = 0, hyper_w = 1, hyper_s = 1):
    mu = np.random.normal(hyper_mu, hyper_s**2, size=(N, n))
    s = np.sqrt(1 + np.random.exponential(1/hyper_w, size=(N, n)))
    y = np.zeros((N, n))
    y[:, :k] = (1 + np.random.exponential(1/hyper_v, size=(N, k)))

    Z_target = np.random.normal(mu + y, s)
    indicators = np.c_[np.zeros(shape=(N, k)) + 1, np.zeros(shape=(N, n-k))]
    
    Z_target = [Z_target, indicators]
    
    Z_decoys = []
    
    for i in range(decoys):
        Z_decoys.append(np.random.normal(mu, s))
    
    return [Z_target, Z_decoys]

if __name__ == "__main__":
    k = 2000
    
    Z_target, Z_decoys = gen_scores(N = 1, k = k)
    
    Z_target = Z_target[0]
    Z_decoy1 = Z_decoys[0]
    Z_decoy2 = Z_decoys[1]
    
    Z_target = Z_target.flatten()
    Z_decoy1 = Z_decoy1.flatten()
    Z_decoy2 = Z_decoy2.flatten()
    
    fig, axs = plt.subplots(5, 1)

    b = np.linspace(-10, 20, 60)

    his = np.histogram(Z_target, bins=b)
    print(his[0])
    up = np.max(his[0])
    
    axs[0].hist(Z_target, bins=b)
    axs[0].vlines(np.mean(Z_target), 0, up, colors = "red")
    axs[1].hist(Z_target[k:], bins=b)
    axs[1].vlines(np.mean(Z_target[k:]), 0, 0.8*up, colors = "red")
    axs[2].hist(Z_target[:k], bins=b)
    axs[2].vlines(np.mean(Z_target[:k]), 0, 0.2*up, colors = "red")
    axs[3].hist(Z_decoy1, bins=b)
    axs[3].vlines(np.mean(Z_decoy1), 0, up, colors = "red")
    axs[4].hist(Z_decoy2, bins=b)
    axs[4].vlines(np.mean(Z_decoy2), 0, up, colors = "red")

    plt.show()