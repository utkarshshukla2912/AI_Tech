from tqdm import tqdm
import pandas as pd
import glob

files = glob.glob('../objects/reconstructed_files/*.csv')
for file in tqdm(files):
    file_name = file[file.rindex('/')+1:]
    df = pd.read_csv(file)
    df = df.drop('Unnamed: 0',axis = 1)
    df = df.T
    df.to_csv('../objects/post_processed/'+file_name,header=False, index=False)
