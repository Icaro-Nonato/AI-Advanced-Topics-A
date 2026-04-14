import wandb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import ks_2samp, chi2_contingency
import yaml, random, matplotlib.pyplot as plt
import kagglehub, shutil, os

wandb.login(key="wandb_v1_NoILfifJBUCFh3NrTHaDXCBARwe")

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

path = kagglehub.dataset_download("prathamtripathi/customersegmentation")

print(f"Baixado em {path}")

destination = os.getcwd()
shutil.copytree(path, os.path.join(destination, "dataset"), dirs_exist_ok=True)

df_raw = pd.read_csv("dataset/Customer_Segmentation.csv", low_memory=False)

print(f"shape: {df_raw.shape}")
print(f"colunas: {list(df_raw.columns[:5])}")