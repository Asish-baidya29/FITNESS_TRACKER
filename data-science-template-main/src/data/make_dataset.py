import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

single_file_acc = pd.read_csv(r"C:\Users\ASISH\OneDrive\Desktop\New folder\data-science-template-main\data-science-template-main\data\raw\MetaMotion\MetaMotion\A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")
single_file_gyr = pd.read_csv(r"C:\Users\ASISH\OneDrive\Desktop\New folder\data-science-template-main\data-science-template-main\data\raw\MetaMotion\MetaMotion\A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files=glob("../../data/raw/MetaMotion/MetaMotion/*.csv")
len(files)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

data_path="../../data/raw/MetaMotion/MetaMotion/"
f=files[0]

participant_name = f.split("-")[0].split("\\")[-1].split("/")[-1]

"""participant
label
category"""

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------


# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------