import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import  joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
df=pd.read_csv("Crop_recommendation.csv")
print(df)
print(df.info())
print(df.shape)
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.columns)
