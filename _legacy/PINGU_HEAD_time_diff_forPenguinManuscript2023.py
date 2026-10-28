import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/path/to/pengu_head_movies/saccade_data/saccades_body.csv')
df2 = pd.read_csv('/path/to/pengu_head_movies/saccade_data/saccades_head.csv')

df['type'] = 'body'
df2['type'] = 'head'

df_saccades = pd.concat((df,df2))
df_saccades.to_csv('/path/to/pengu_head_movies/saccade_data/saccades_combi.csv',index=False)


df = df_saccades

# Group by 'id'
grouped = df.groupby('id')

# Initialize an empty list to store the results
time_differences = []

# Loop through each group (i.e., each movie)
for name, group in grouped:
    # Separate the data into head and body movements
    head_saccades = group[group['type'] == 'head']
    body_saccades = group[group['type'] == 'body']

    # Loop through each direction ('left', 'right')
    for direction in ['left', 'right']:
        head_saccades_dir = head_saccades[head_saccades['direction'] == direction]
        body_saccades_dir = body_saccades[body_saccades['direction'] == direction]

        # Loop through each head saccade and find the closest body saccade in time
        for index, head_saccade in head_saccades_dir.iterrows():
            head_time = head_saccade['saccade_peak_s']
            time_diffs = (body_saccades_dir['saccade_peak_s'] - head_time).abs()

            if len(time_diffs) > 0:
                # Find the closest body saccade in time
                closest_body_saccade_idx = time_diffs.idxmin()
                closest_body_saccade_time = body_saccades_dir.loc[closest_body_saccade_idx, 'saccade_peak_s']

                # Calculate the time difference
                time_difference = closest_body_saccade_time - head_time

                # Store the results
                time_differences.append({
                    'id': name,
                    'direction': direction,
                    'head_time': head_time,
                    'body_time': closest_body_saccade_time,
                    'time_difference': time_difference
                })

# Convert the list of time differences to a DataFrame for easier analysis
time_differences_df = pd.DataFrame(time_differences)
# Filter out time differences larger than 1 second
filtered_time_differences_df = time_differences_df[time_differences_df['time_difference'].abs() <= 1]

g = sns.displot(filtered_time_differences_df, x="time_difference", stat="probability", bins= np.linspace(-1,1,11))
# Add arrows and text annotations to the plot
plt.annotate('Body Faster', xy=(-0.95, 0.1), xytext=(-0.8, 0.1),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             horizontalalignment='left', verticalalignment='bottom')

plt.annotate('Head Faster', xy=(0.95, 0.1), xytext=(0.8, 0.1),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             horizontalalignment='right', verticalalignment='bottom')
g.fig.suptitle(f'head saccade timing differnces(same direction | n = {len(filtered_time_differences_df)})')
g.savefig('/path/to/pengu_head_movies/figures/time_diff.svg')
g.savefig('/path/to/pengu_head_movies/figures/time_diff.png')




# Initialize an empty list to store the results
time_differences = []

# Loop through each group (i.e., each movie)
for name, group in grouped:
    # Separate the data into head and body movements
    head_saccades = group[group['type'] == 'head']
    body_saccades = group[group['type'] == 'body']
    # No longer filtering by direction
    head_saccades_dir = head_saccades
    body_saccades_dir = body_saccades

    # Loop through each combination of different directions
    for head_direction, body_direction in [('left', 'right'), ('right', 'left')]:
        head_saccades_dir = head_saccades[head_saccades['direction'] == head_direction]
        body_saccades_dir = body_saccades[body_saccades['direction'] == body_direction]

  


        # Loop through each head saccade and find the closest body saccade in time
        for index, head_saccade in head_saccades_dir.iterrows():
            head_time = head_saccade['saccade_peak_s']
            time_diffs = (body_saccades_dir['saccade_peak_s'] - head_time).abs()

            if len(time_diffs) > 0:
                # Find the closest body saccade in time
                closest_body_saccade_idx = time_diffs.idxmin()
                closest_body_saccade_time = body_saccades_dir.loc[closest_body_saccade_idx, 'saccade_peak_s']

                # Calculate the time difference
                time_difference = closest_body_saccade_time - head_time

                # Store the results
                time_differences.append({
                    'id': name,
                    'direction': direction,
                    'head_time': head_time,
                    'body_time': closest_body_saccade_time,
                    'time_difference': time_difference
                })

# Convert the list of time differences to a DataFrame for easier analysis
time_differences_df = pd.DataFrame(time_differences)
# Filter out time differences larger than 1 second
filtered_time_differences_df = time_differences_df[time_differences_df['time_difference'].abs() <= 1]

h = sns.displot(filtered_time_differences_df, x="time_difference", stat="probability", bins= np.linspace(-1,1,11))
# Add arrows and text annotations to the plot
plt.annotate('Body Faster', xy=(-0.95, 0.1), xytext=(-0.8, 0.1),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             horizontalalignment='left', verticalalignment='bottom')

plt.annotate('Head Faster', xy=(0.95, 0.1), xytext=(0.8, 0.1),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             horizontalalignment='right', verticalalignment='bottom')
h.fig.suptitle(f'head saccade timing differnces (diff. direction | n = {len(filtered_time_differences_df)})')
h.savefig('/path/to/pengu_head_movies/figures/time_diff_inv.svg')
h.savefig('/path/to/pengu_head_movies/figures/time_diff_inv.png')
plt.show()
print(time_differences_df)




plt.show()
print(time_differences_df)
