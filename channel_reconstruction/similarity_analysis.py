import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob

original_files = glob.glob('../data/*.csv')
reconstructed_files = glob.glob('../objects/reconstructed_files/*.csv')
columns = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'P4', 'FC5', 'FC1', 'FC2', 'FC4', 'CP5', 'CP1', 'CP2', 'CP4','Label']
columns_to_hide = ["F3", "Fz", "F4", "C3", "Cz", "C4", "FC5", "FC4", "CP5", "CP4"]
columns_to_keep = ['P3', 'P4', 'FC1', 'FC2', 'CP1', 'CP2']
original_files.sort()
reconstructed_files.sort()

master_similarity = []
for file_pair in tqdm(zip(original_files,reconstructed_files)):
    file_similarity = []
    original = pd.read_csv(file_pair[0],header=None)
    original = original.T.astype(float)
    original.columns = columns
    reconstructed = pd.read_csv(file_pair[1])

    for col in columns_to_hide:
        channel_ori = original[col].values.reshape(1,-1)
        channel_rec = reconstructed[col].values.reshape(1,-1)
        file_similarity.append(cosine_similarity(channel_ori, channel_rec)[0][0])
    master_similarity.append(sum(file_similarity))

print(file_similarity)
