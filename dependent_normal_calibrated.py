import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gen_scores(N = 1000, n = 10000, k=2000, decoys = 2, hyper_v = 0.35, hyper_mu = 0, hyper_w = 1, hyper_s = 1):
    pi1 = k/n
    
    nn = int(n * pi1)
    nf = int(n * (1 - pi1))
    
    Ddf = pd.read_csv("cleandata/Ddf.csv")
    Sdf = pd.read_csv("cleandata/Sdf.csv")
    
    # Step 4: Sampling native and foreign peptide-charge groups
    
    md = Ddf
    mt = Sdf
    
    md = md.sample(frac=1).reset_index(drop=True)
    mt = mt.sample(frac=1).reset_index(drop=True)
    
    smd = md.iloc[np.random.choice(md.index, size=nf)].reset_index(drop=True) # Do we want the same dependence structure for each simulation? 
    smt = mt.iloc[np.random.choice(mt.index, size=nn)].reset_index(drop=True) # Actually yes otherwise mismatched data shape
    
    kd = np.min(np.arange(len(smd))[np.cumsum(smd["multiplicity"]) >= nf])
    kt = np.min(np.arange(len(smt))[np.cumsum(smt["multiplicity"]) >= nn])
    
    smd = smd[np.arange(len(smd)) <= kd]
    smt = smt[np.arange(len(smt)) <= kt]
    
    # Step 5: Generating Scores
    
    mu = np.zeros((N, len(smt) + len(smd)))
    s = np.ones((N, len(smt) + len(smd)))
    y = 1 + np.random.exponential(scale=1/hyper_v, size=(N, len(smt)))
    
    Xnat = np.random.normal(mu[:, :len(smt)] + y, s[:, :len(smt)])
    Xnat = np.repeat(Xnat, repeats=smt["multiplicity"], axis=1)
    
    Ynat = np.random.normal(mu[:, :len(smt)], s[:, :len(smt)])
    Ynat = np.repeat(Ynat, repeats=smt["multiplicity"], axis=1)
    
    Dnats = []
    for i in range(decoys):
        Dnats.append(np.repeat(np.random.normal(mu[:, :len(smt)], s[:, :len(smt)]), repeats=smt["multiplicity"], axis=1))
        
    Xfor = -100 * np.ones((N, len(smd)))
    Xfor = np.repeat(Xfor, repeats=smd["multiplicity"], axis=1)
    
    Yfor = np.random.normal(mu[:, len(smt):], s[:, len(smt):])
    Yfor = np.repeat(Yfor, smd["multiplicity"], axis=1)
    
    Dfors = []
    for i in range(decoys):
        Dfors.append(np.repeat(np.random.normal(mu[:, len(smt):], s[:, len(smt):]), repeats=smd["multiplicity"], axis=1))
        
    # Step 6: Introducing experimental noise
    
    def gen_noise(l, d, v):
        for i in range(len(d)):
            l.append(np.random.normal(size=(N, np.int16(d["multiplicity"][i]))) * d[v][i])
    
    # Native noise
    
    noise = []
    gen_noise(noise, smt, "sdt")
    noise = np.concat(noise, axis=1)
    
    Xnat = Xnat + noise
    
    noise = []
    gen_noise(noise, smt, "sdd")
    
    dnoise = [[] for _ in range(decoys)]
    for i in range(decoys):
        gen_noise(dnoise[i], smt, "sdd")

    noise = np.concat(noise, axis=1)
    for i in range(decoys):
        dnoise[i] = np.concat(dnoise[i], axis=1)
        
    Ynat = Ynat + noise
    for i in range(decoys):
        Dnats[i] = Dnats[i] + dnoise[i]
        
    # Foreign noise
    
    noise = []
    gen_noise(noise, smd, "sdd")
    
    dnoise = [[] for _ in range(decoys)]
    for i in range(decoys):
        gen_noise(dnoise[i], smd, "sdd")
            
    noise = np.concat(noise, axis=1)
    for i in range(decoys):
        dnoise[i] = np.concat(dnoise[i], axis=1)
        
    Yfor = Yfor + noise
    for i in range(decoys):
        Dfors[i] = Dfors[i] + dnoise[i]
    
    # Step 7: Determining simulated target scores
    
    Z_target = [np.max((np.c_[Xnat, Xfor], np.c_[Ynat, Yfor]), axis=0)]
    Z_target.append((np.c_[Xnat, Xfor] > np.c_[Ynat, Yfor]) * 1)
    
    Z_decoys = []
    for i in range(decoys):
        Z_decoys.append(np.c_[Dnats[i], Dfors[i]])
        
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

    b = np.linspace(-5, 15, 60)

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