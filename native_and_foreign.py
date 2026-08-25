import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
start_time = time.process_time()

target_df = pd.read_csv("cleandata/target_data.csv")
decoy_df = pd.read_csv("cleandata/decoy_data.csv")

# Step 1: Extracting peptide-charge groups

target_identifiers = np.apply_along_axis(lambda x: str(x[0]) + " " + str(x[1]), 1, target_df.iloc[:, 3:]) # This is somehow fast
decoy_identifiers = np.apply_along_axis(lambda x: str(x[0]) + " " + str(x[1]), 1, decoy_df.iloc[:, 3:])

intersect = np.intersect1d(target_identifiers, decoy_identifiers)

present = np.isin(target_identifiers, intersect)

target_df["identifier"] = target_identifiers
decoy_df["identifier"] = decoy_identifiers

target_df = target_df.sort_values(by="identifier")
decoy_df = decoy_df.sort_values(by="identifier")

target_df.index = range(len(target_df))
decoy_df.index = range(len(decoy_df))

merged_df = target_df[present]
merged_df.columns = ["p", "c", "s_t", "f", "sp", "id"]
merged_df.sort_values(by="p")

present2 = np.isin(decoy_identifiers, intersect)
merged_df.insert(3, "s_d", decoy_df.iloc[present, 2])

# Step 2: Grouping by peptide-charge pairs

# T_df

merged_df["group"] = merged_df.iloc[:, :2].apply(lambda x: ",".join([str(y) for y in x]), axis=1)

multiplicity = np.unique(merged_df["group"], return_counts=True)

# plt.hist(multiplicity[1], bins=40)
# plt.show()

grouped_dict = {}

for i in range(len(merged_df)):
    k = merged_df.iloc[i, -1]
    
    if k in grouped_dict:
        grouped_dict[k].append(list(merged_df.iloc[i, 2:4]))
    else:
        grouped_dict[k] = [list(merged_df.iloc[i, 2:4])]

for k, v in grouped_dict.items():
    grouped_dict[k] = np.array(v)

Tdf = pd.DataFrame()
Tdf["group"] = multiplicity[0]
Tdf["multiplicity"] = multiplicity[1]
Tdf["s"] = [np.max(grouped_dict[g][:, 0]) for g in multiplicity[0]]
Tdf["sdt"] = [np.std(grouped_dict[g][:, 0]) for g in multiplicity[0]]
Tdf["sdd"] = [np.std(grouped_dict[g][:, 1]) for g in multiplicity[0]]

print("T_df:\n", Tdf)

# D_df

decoy_df["group"] = decoy_df.iloc[:, :2].apply(lambda x: ",".join([str(y) for y in x]), axis=1)

multiplicity = np.unique(decoy_df["group"], return_counts=True)

# plt.hist(multiplicity[1], bins=40)
# plt.show()

grouped_dict = {}

for i in range(len(merged_df)):
    k = decoy_df.iloc[i, -1]
    
    if k in grouped_dict:
        grouped_dict[k].append(decoy_df.iloc[i, 2])
    else:
        grouped_dict[k] = [decoy_df.iloc[i, 2]]

for k, v in grouped_dict.items():
    grouped_dict[k] = np.array(v)

Ddf = pd.DataFrame()
Ddf["group"] = multiplicity[0]
Ddf["multiplicity"] = multiplicity[1]
Ddf["s"] = [np.max(grouped_dict[g]) for g in multiplicity[0]]
Ddf["sdd"] = [np.std(grouped_dict[g]) for g in multiplicity[0]]

print("D_df:\n", Ddf)

# Step 3: Selecting representative native and foreign groups

Tdf["indicator"] = np.ones(len(Tdf))
Ddf["indicator"] = np.ones(len(Ddf)) * -1

S = pd.concat([Tdf.iloc[:, [0, 1, 2, 3, 4, -1]], Ddf.iloc[:, [0, -1, 2]]])
idxs = np.random.permutation(len(S))
S = S.sample(frac=1).reset_index(drop=True)
S = S.sort_values(by="s", ascending=False)

T = np.cumsum(S["indicator"] == 1)
D = np.cumsum(S["indicator"] == -1)

FDR = D / np.clip(T, 1, None)
alpha = 0.25

k = np.max(np.arange(1, len(T)+1) * (FDR <= alpha))
print(f"k* was selected at {k}")

# # Curious!

# plt.plot(range(len(T)), T/(T+D))
# plt.plot(range(len(T)), S["s"])
# plt.plot(range(len(T)), FDR)
# plt.axhline(0.25, 0, len(T))

# plt.show()

Sdf = S[(np.arange(1, len(T)+1) < k) & (S["indicator"] == 1)]
print(Sdf)

Sdf.to_csv("cleandata/Sdf.csv", index=False)
Ddf.to_csv("cleandata/Ddf.csv", index=False)

print(f"Program executed in: {time.process_time() - start_time:.3f} seconds.")