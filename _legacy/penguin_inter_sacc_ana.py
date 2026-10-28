import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

# Statistics Function

def compute_stats(group):
    durations = group['saccade_duration_msec']
    n = len(durations)
    median = np.median(durations)
    mean = np.mean(durations)
    sem = scipy.stats.sem(durations)
    max_dur =np.max(durations)
    if n > 1:
        ci_low, ci_high = scipy.stats.t.interval(0.95, n-1, loc=mean, scale=sem)
    else:
        # For n=1, use mean ± sem
        ci_low, ci_high = mean - sem, mean + sem
    return pd.Series({
        'Median Dur (ms)': median,
        'Lower 95% CI': ci_low,
        'Upper 95% CI': ci_high,
        'Mean Dur (ms)': mean,
        'Max. Dur (ms)': max_dur,
        'SEM': sem
    })


# Set frames per second
fps = 25

# Read CSV files into DataFrames
df_dataset2_body = pd.read_csv('/path/to/Penguin_Rostock/pengu_head_movies/saccade_data/saccades_body.csv' )
df_dataset2_combi = pd.read_csv('/path/to/Penguin_Rostock/pengu_head_movies/saccade_data/saccades_combi.csv' )
df_dataset2_head = pd.read_csv('/path/to/Penguin_Rostock/pengu_head_movies/saccade_data/saccades_head.csv')
df_dataset1_inter_sacc = pd.read_csv('/path/to/Penguin_Rostock/penguins/body_inter_saccade_dur.csv')
df_dataset1_sacc = pd.read_csv('/path/to/Penguin_Rostock/penguins/body_saccade_dur.csv')

# Initialize lists to collect data
species          = list()
dataset          = list()
saccade_type     = list() # 'saccade' or 'intersaccade'
movement_type    = list() # 'free', 'associated', 'intersaccade'
saccade_duration = list()
bodypart         = list()


for name,movement_type_name,df in [('body','free',df_dataset2_body),
                                   ('head','free',df_dataset2_head),
                                   ('head+body','associated',df_dataset2_combi)]:  
    # Extract saccade durations in milliseconds
    temp_sacc_dur = df.saccade_duration_ms.to_list()
    temp_sacc_type = ['saccade' for i in temp_sacc_dur]
    temp_movement_type = [movement_type_name for _ in temp_sacc_dur]

    # Compute intersaccadic durations between consecutive saccades
    temp_intersacc_dur = np.diff(np.vstack((df.saccade_start_idx.to_numpy()[1::],df.saccade_stop_idx.to_numpy()[0:-1])).T,axis=1)
    temp_intersacc_dur = temp_intersacc_dur[temp_intersacc_dur>0]/fps*1000
    temp_sacc_type += ['intersaccade' for i in temp_intersacc_dur]
    temp_movement_type += ['intersaccade' for _ in temp_intersacc_dur]
    
    # Extend the main lists with data from this iteration
    saccade_duration += temp_sacc_dur + temp_intersacc_dur.flatten().tolist()
    saccade_type += temp_sacc_type
    movement_type += temp_movement_type
    species += ['gentoo' for i in temp_sacc_type]
    dataset += [2 for i in temp_sacc_type] 
    bodypart += [name for i in temp_sacc_type] 


# Process Dataset 1 (open environment) - Saccades
temp_dur = list(df_dataset1_sacc.sacc_dur_sec.to_numpy()*1000)
saccade_duration += temp_dur
saccade_type +=['saccade' for i in temp_dur] 
movement_type += ['saccade' for _ in temp_dur]   
species += df_dataset1_sacc.species.to_list()
dataset += [1 for i in temp_dur] 
bodypart += ['body' for i in temp_dur] 

# Process Dataset 1 (open environment) - Intersaccadic intervals
temp_dur = list(df_dataset1_inter_sacc.inter_sacc_dur_sec.to_numpy()*1000)
saccade_duration += temp_dur
saccade_type +=['intersaccade' for i in temp_dur] 
movement_type += ['intersaccade' for _ in temp_dur]   
species += df_dataset1_inter_sacc.species.to_list()
dataset += [1 for i in temp_dur] 
bodypart += ['body' for i in temp_dur] 


# Ensure all lists are of the same length
for x in (saccade_duration, saccade_type, species, dataset, bodypart):
    print(len(x))

# Create DataFrame from the collected data
df_durations = pd.DataFrame({
    'saccade_duration_msec': saccade_duration,
    'saccade_type': saccade_type,
    'movement_type': movement_type,
    'species': species,
    'dataset': dataset,
    'bodypart': bodypart
})

df_durations['id'] = [f'{ds} {bp} {s}' for ds,bp,s in list(zip(dataset,bodypart,species))]
df_durations.to_csv('/path/to/Penguin_Rostock/comb_durations.csv',index=False)


# Compute the statistics for the saccade durations
group_cols = ['dataset', 'species', 'movement_type', 'bodypart']
grouped = df_durations.groupby(group_cols)
# Apply the compute_stats function to each group
stats_df = grouped.apply(compute_stats).reset_index()

# Rearranging columns
stats_df = stats_df[['dataset', 'species', 'movement_type', 'bodypart',
                     'Median Dur (ms)', 'Lower 95% CI', 'Upper 95% CI', 'Mean Dur (ms)',"Max. Dur (ms)", 'SEM']]

# Save the statistics table to a CSV file
stats_df.to_csv('/path/to/Penguin_Rostock/saccade_statistics.csv', index=False)

sns.boxplot(data=df_durations, x="id", y="saccade_duration_msec",hue='saccade_type')
plt.yscale('log')
plt.show()

print('done')

