import numpy as np
import matplotlib.pyplot as plt

def gen_scores(N = 1000, n = 10000, k = 2000, decoys = 2, a = 0.05, b = 3, c = 100):
    X = np.c_[1 - np.random.beta(a, b, size=(N, k)), -np.ones((N, n-k))]
    Y = np.c_[np.random.beta(c-1, 1, size=(N, k)), np.random.beta(c, 1, size=(N, n-k))]
    
    Z_target = [np.max([X, Y], axis=0)]
    Z_target.append((X > Y)*1) # No, you checked this already. It's right even though it looks wrong
    
    Z_decoys = []
    
    for i in range(decoys):
        Z_decoys.append(np.random.beta(c, 1, size=(N, n)))
        
    return [Z_target, Z_decoys]
    
if __name__ == "__main__":
    Z_target, Z_decoys = gen_scores(N=1)
    
    k = 2000
    
    Z_target = Z_target[0]
    Z_decoy1 = Z_decoys[0]
    Z_decoy2 = Z_decoys[1]
    
    Z_target = Z_target.flatten()
    Z_decoy1 = Z_decoy1.flatten()
    Z_decoy2 = Z_decoy2.flatten()
    
    fig, axs = plt.subplots(1, 5, figsize=(15, 7))

    b = np.linspace(0.97, 1, 100)

    print(Z_target[:k])

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
    