import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction



# Load data
# -----------

df = pd.read_pickle("../../data/interim/01_outliers_removed_chauvenets.pkl")
predictor_col = list(df.columns[:6])

#plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"]=(20,5)
plt.rcParams["figure.dpi"]=100
plt.rcParams["lines.linewidth"]=2

# Dealing with missing values (imputation)
# --------------------------------------------

for col in predictor_col:
    df[col]=df[col].interpolate()
    
df.info()


# Calculating set duration
# -------------------------------

df[df["set"]==25]["acc_y"].plot()

duration = df[df["set"]==1].index[-1] - df[df["set"]==1].index[0]
duration.seconds

for s in df["set"].unique():
    
    start = df[df["set"]==s].index[0]
    stop = df[df["set"]==s].index[-1]
    
    duration= stop-start
    df.loc[(df["set"]==s),"duration"]=duration.seconds
    
duration.df = df.groupby(["category"])["duration"].mean()

duration.df.iloc[0]/5
duration.df.iloc[1]/10


# Butterworth lowpass filter
# ----------------------------------

df_lowpass=df.copy()
LowPass=LowPassFilter()

fs = 1000/200 #epoch frequency is 200 ms
cutoff = 1.3

df_lowpass=LowPass.low_pass_filter(df_lowpass,"acc_y",fs,cutoff,order=5)

subset = df_lowpass[df_lowpass["set"]==67]
print(subset["lable"][0])

fig,ax=plt.subplots(nrows=2,sharex=True,figsize=(20,10))
ax[0].plot(subset["acc_y"].reset_index(drop=True),label="raw_data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True),label="butterworth filtter")
ax[0].legend(subset["upper center"],bbox_to_anchor=(0.5,1.15),fancybox=True,shadow=True)
ax[1].legend(subset["upper center"],bbox_to_anchor=(0.5,1.15),fancybox=True,shadow=True)

for col in predictor_col:
    df_lowpass = LowPass.low_pass_filter(df_lowpass,col,fs,cutoff,order=5)
    df_lowpass[col]=df_lowpass[col+"_lowpass"]
    del df_lowpass[col+"_lowpass"]


# Principal component analysis PCA
# -------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

pc_values=PCA.determine_pc_explained_variance(df_pca,predictor_col)

#ploting pc_values on graph
plt.figure(figsize=(10,10))
plt.plot(range(1,len(predictor_col)+1),pc_values, marker="o")
plt.xlabel("principal components number")
plt.ylabel("explained variance")
plt.show()

df_pca = PCA.apply_pca(df_pca,predictor_col,number_comp=3)

subset = df_pca[df_pca["set"]==35]
subset[["pca_1","pca_2","pca_3"]].plot()


# Sum of squares attributes
# -------------------------------

df_square = df_pca.copy()
acc_r = df_square["acc_x"]**2+df_square["acc_y"]**2+df_square["acc_z"]**2
gyr_r = df_square["gyr_x"]**2+df_square["gyr_y"]**2+df_square["gyr_z"]**2

df_square["acc_r"]=np.sqrt(acc_r)
df_square["gyr_r"]=np.sqrt(gyr_r)

df_square


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------