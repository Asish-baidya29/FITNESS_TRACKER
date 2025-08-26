import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


df = pd.read_pickle("../../data/interim/DataProcessed.pkl")


# Plot single columns
#------------------------------------

set_df = df[df["set"]==1]
plt.plot(set_df["acc_y"])

plt.plot(set_df["acc_y"].reset_index(drop=True))


# Plot all exercises
#------------------------------------

for lable in df["lable"].unique():
    subset=df[df["lable"]==lable]
    fig,ax =plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True),label=lable)
    plt.legend()
    plt.show()
    
# Plot small data to visualize graph patterns
for lable in df["lable"].unique():
    subset=df[df["lable"]==lable]
    fig,ax =plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True),label=lable)
    plt.legend()
    plt.show()    


# Adjust plot settings
#------------------------------------

mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"]=(20,5)
mpl.rcParams["figure.dpi"]=100


# Compare medium vs. heavy sets
#------------------------------------

category_df = df.query("lable=='bench'").query("participant=='A'").reset_index()

fig,ax =plt.subplots()
category_df.groupby(["category"])["acc_y"].plot() #groupby heavy & medium
ax.set_ylabel("acc-y")
ax.set_xlabel("sample")
plt.legend()

for labels in df["lable"].unique():
    
    category_df = df.query("lable==@labels").query("participant=='A'").reset_index()
    
    fig,ax =plt.subplots()
    category_df.groupby(["category"])["acc_y"].plot() #groupby heavy & medium
    ax.set_ylabel("acc-y")
    ax.set_xlabel("sample")
    plt.legend()
    print(labels)


# Compare participants
#------------------------------------

participant_df = df.query("lable=='bench'").sort_values("participant").reset_index()

fig,ax =plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot() #groupby participant
ax.set_ylabel("acc-y")
ax.set_xlabel("sample")
plt.legend()


# Plot multiple axis
# -------------------------------------

lable="squat"
participant ="A"
all_axis_df = df.query(f"lable == '{lable}'").query(f"participant=='{participant}'").reset_index()

fig,ax =plt.subplots()
all_axis_df[["acc_x","acc_y","acc_z"]].plot(ax=ax)
ax.set_xlabel("sample")
ax.set_ylabel("acc_y")
plt.legend()


# Create a loop to plot all combinations per sensor
# --------------------------------------------------------

lables=df["lable"].unique()
participants =df["participant"].unique()

for lable in lables:
    for participant in participants:
        all_axis_df = df.query("lable == @lable").query("participant==@participant").reset_index()

        if len(all_axis_df)>0:
            fig,ax =plt.subplots()
            all_axis_df[["acc_x","acc_y","acc_z"]].plot(ax=ax)
            ax.set_xlabel("sample")
            ax.set_ylabel("acc_y")
            plt.title(f"{lable}({participant})".title())
            plt.legend()


for lable in lables:
    for participant in participants:
        all_axis_df = df.query("lable == @lable").query("participant==@participant").reset_index()

        if len(all_axis_df)>0:
            fig,ax =plt.subplots()
            all_axis_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax)
            ax.set_xlabel("sample")
            ax.set_ylabel("gyr_y")
            plt.title(f"{lable}({participant})".title())
            plt.legend()


# Combine plots in one figure
# ---------------------------------

lable="row"
participant ="A"
combined_plot_df = df.query(f"lable == '{lable}'").query(f"participant=='{participant}'").reset_index(drop=True)

fig,ax = plt.subplots(nrows=2,sharex=True,figsize=(20,10))
combined_plot_df[["acc_x","acc_y","acc_z"]].plot(ax=ax[0])
combined_plot_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax[1])

ax[0].legend(loc="upper center",bbox_to_anchor=(0.5,1.15),ncol=3,fancybox=True,shadow=True)
ax[1].legend(loc="upper center",bbox_to_anchor=(0.5,1.15),ncol=3,fancybox=True,shadow=True)
ax[1].set_xlabel("sample")

# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

lables=df["lable"].unique()
participants =df["participant"].unique()

for lable in lables:
    for participant in participants:
        all_axis_df = df.query("lable == @lable").query("participant==@participant").reset_index()

        if len(combined_plot_df)>0:
            fig,ax = plt.subplots(nrows=2,sharex=True,figsize=(20,10))
            combined_plot_df[["acc_x","acc_y","acc_z"]].plot(ax=ax[0])
            combined_plot_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax[1])

            ax[0].legend(loc="upper center",bbox_to_anchor=(0.5,1.15),ncol=3,fancybox=True,shadow=True)
            ax[1].legend(loc="upper center",bbox_to_anchor=(0.5,1.15),ncol=3,fancybox=True,shadow=True)
            ax[1].set_xlabel("sample")
            
            plt.savefig(f"../../reports/figures/{lable.title()}({participant}).png")
            plt.show()