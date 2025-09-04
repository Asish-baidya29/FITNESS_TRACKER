import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation


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



# Temporal abstraction
# --------------------------

df_temporal = df_square.copy()

NumAbs=NumericalAbstraction()

predictor_col=predictor_col+["acc_r","gyr_r"]

ws = int (1000/200)

for col in predictor_col:
    df_temporal=NumAbs.abstract_numerical(df_temporal,[col],ws,"mean")
    df_temporal=NumAbs.abstract_numerical(df_temporal,[col],ws,"std")

df_temporal_list =[]
for s in df_temporal["set"].unique():
    subset=df_temporal[df_temporal["set"]==s].copy()
    for col in predictor_col:
        subset=NumAbs.abstract_numerical(subset,[col],ws,"mean")
        subset=NumAbs.abstract_numerical(subset,[col],ws,"std")
    df_temporal_list.append(subset)
df_temporal = pd.concat(df_temporal_list)

df_temporal.info()

subset[["acc_y","acc_y_temp_mean_ws_5","acc_y_temp_std_ws_5"]].plot()

# Frequency features
# --------------------------

df_freq = df_temporal.copy().reset_index()

FreqAbs = FourierTransformation()
fs = int (1000/200)
ws = int(2800/200)

df_freq=FreqAbs.abstract_frequency(df_freq,["acc_y"],ws,fs)

df_freq_list=[]
for s in df_freq["set"].unique():
    print(f"Applying Fourier transformation to set {s}")
    subset = df_freq[df_freq["set"]==s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset,predictor_col,ws,fs)
    df_freq_list.append(subset)
    
df_freq=pd.concat(df_freq_list).set_index("epoch (ms)",drop=True)

# Dealing with overlapping windows
# --------------------------------------

df_freq=df_freq.dropna()

df_freq=df_freq.iloc[::2] # skip every 2 row

# Clustering
# ----------------

from sklearn.cluster import KMeans

df_cluster = df_freq.copy()
cluster_col = ["acc_x", "acc_y", "acc_z"]

subset = df_cluster[cluster_col]   # only once
k_values = range(2, 10)
inertias = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

# plot elbow graph 
plt.figure(figsize=(10,10))
plt.plot(k_values,inertias)
plt.xlabel("k")
plt.ylabel("sum of squared distances")
plt.show()

# our k =5 
kmeans = KMeans(n_clusters=5,n_init=20,random_state=0)
subset = df_cluster[cluster_col]
df_cluster["cluster"]= kmeans.fit_predict(subset)

df_cluster["cluster"].unique()


# Plot clusters in 3D
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")

for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=f"Cluster {c}")

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()


# Plot accelerometer data by label (for comparison)

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")

for l in df_cluster["lable"].unique():
    subset = df_cluster[df_cluster["lable"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()


# Export dataset
# ------------------------

df_cluster.to_pickle("../../data/interim/02_data_features.pkl")