import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = []

# with open("data/narrow_sep_0_0.tide-search.decoy.txt") as f:
with open("data/narrow_sep_0_0.tide-search.target.txt") as f:
    l = f.readline()
    
    while l:
        data.append(l)
        l = f.readline()

dict = {}
cols = []

for n in data[0].split("\t"):
    dict[n] = []
    cols.append(n)

for d in data[1:]:
    for i, v in enumerate(d.split("\t")):
        dict[cols[i]].append(v)

df = pd.DataFrame(dict)

df = df[df["xcorr rank"] == "1"]

df = df.iloc[:, [12, 2, 8, 0, 1]]

df["charge"] = df["charge"].astype("float")
df["xcorr score"] = df["xcorr score"].astype("float")
df["scan"] = df["scan"].astype("float")

# df.to_csv("cleandata/decoy_data.csv", index=False)
df.to_csv("cleandata/target_data.csv", index=False)