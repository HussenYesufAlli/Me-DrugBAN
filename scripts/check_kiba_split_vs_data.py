# import pickle
# import json
# import numpy as np

# # Open KIBA affinity matrix
# Y = pickle.load(open("data/kiba/Y", "rb"), encoding="latin1")
# print("Y shape:", Y.shape)
# print("Num proteins:", Y.shape[0])
# print("Num ligands:", Y.shape[1])
# label_row_inds, label_col_inds = np.where(~np.isnan(Y))
# print("Valid Y entries:", len(label_row_inds))

# with open("data/kiba/folds/train_fold_setting1.txt") as f:
#     indices = eval(f.read().strip())
# if isinstance(indices, list) and len(indices) == 1 and isinstance(indices[0], list):
#     indices = indices[0]  # handle nesting
# if isinstance(indices[0], list):
#     indices = indices[0]

# print("Max index in split:", max(indices), "out of", len(label_row_inds)-1)
# print("Min index in split:", min(indices))
# print("Num indices in split:", len(indices))
# # Show a few indices
# for idx in indices[:10]:
#     print(f"Flat index {idx}: protein={label_row_inds[idx]}, ligand={label_col_inds[idx]}")

import pickle
import numpy as np

Y = pickle.load(open("data/kiba/Y", "rb"), encoding="latin1")
print("Before:", Y.shape)
Y = Y.T
print("After:", Y.shape)
with open("data/kiba/Y_fixed", "wb") as f:
    pickle.dump(Y, f)