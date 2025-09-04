import pandas as pd
from glob import glob


single_file_acc = pd.read_csv(r"C:\Users\ASISH\OneDrive\Desktop\New folder\data-science-template-main\data-science-template-main\data\raw\MetaMotion\MetaMotion\A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")
single_file_gyr = pd.read_csv(r"C:\Users\ASISH\OneDrive\Desktop\New folder\data-science-template-main\data-science-template-main\data\raw\MetaMotion\MetaMotion\A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")

# --------------------------------------------------------------
# main function
# --------------------------------------------------------------

files=glob("../../data/raw/MetaMotion/MetaMotion/*.csv")

def read_data_from_files(files):
        
    acc_df=pd.DataFrame()
    gyr_df=pd.DataFrame()

    acc_set=1
    gyr_set=1
    
    # Extract features from filename
    
    for f in files:
        participant = f.split("-")[0].split("\\")[-1].split("/")[-1]
        lable = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
            
        df = pd.read_csv(f)
            
        df["participant"] =participant
        df["lable"] =lable
        df["category"] =category
            
        if "Accelerometer" in f:
            df["set"]=acc_set
            acc_set+=1
            acc_df = pd.concat([acc_df,df])
        elif "Gyroscope" in f:
            df["set"]=gyr_set
            gyr_set+=1
            gyr_df=pd.concat([gyr_df,df])
    
    # Working with datetimes
              
    acc_df.index= pd.to_datetime(acc_df["epoch (ms)"],unit="ms")
    gyr_df.index= pd.to_datetime(gyr_df["epoch (ms)"],unit="ms")

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"] 
    
    return acc_df,gyr_df  

acc_df,gyr_df  = read_data_from_files(files)  


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

final_data=pd.concat([acc_df.iloc[:,:3],gyr_df],axis=1)
final_data.columns=[
    "acc_x",
    "acc_y",
    "acc_z",
    
    "gyr_x",
    "gyr_y",
    "gyr_z",
    
    "participant",
    "lable",
    "category",
    "set",
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

sampling ={
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    
    "participant":"last",
    "lable":"last",
    "category":"last",
    "set":"last"
}
days=[g for n,g in final_data.groupby(pd.Grouper(freq="D"))]
final_resampled_data=pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

final_resampled_data.info()
final_resampled_data["set"]=final_resampled_data["set"].astype("int")

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

final_resampled_data.to_pickle("../../data/interim/DataProcessed.pkl")