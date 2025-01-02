import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_hits_year(year1, year2):
    # Load the dataset
    data = pd.read_csv('dataset.csv')
    data = data[data['Number of Weeks On Top'] > 0]


    songs_y1 = data[data['Year'] == year1]
    songs_y2 = data[data['Year'] == year2]

    print(f"Songs in {year1}: {len(songs_y1)}")
    print(f"Songs in {year2}: {len(songs_y2)}")
    print("\n")
    print(f"Average Danceability in {year1}: {songs_y1['danceability'].mean()}")
    print(f"Average Danceability in {year2}: {songs_y2['danceability'].mean()}")
    print("\n")
    print(f"Average Energy in {year1}: {songs_y1['energy'].mean()}")
    print(f"Average Energy in {year2}: {songs_y2['energy'].mean()}")
    print("\n")
    print(f"Average Loudness in {year1}: {songs_y1['loudness'].mean()}")
    print(f"Average Loudness in {year2}: {songs_y2['loudness'].mean()}")
    print("\n")
    print(f"Average Speechiness in {year1}: {songs_y1['speechiness'].mean()}")
    print(f"Average Speechiness in {year2}: {songs_y2['speechiness'].mean()}")
    print("\n")
    print(f"Average Acousticness in {year1}: {songs_y1['acousticness'].mean()}")
    print(f"Average Acousticness in {year2}: {songs_y2['acousticness'].mean()}")
    print("\n")
    print(f"Average Instrumentalness in {year1}: {songs_y1['instrumentalness'].mean()}")
    print(f"Average Instrumentalness in {year2}: {songs_y2['instrumentalness'].mean()}")
    print("\n")
    print(f"Average Valence in {year1}: {songs_y1['valence'].mean()}")
    print(f"Average Valence in {year2}: {songs_y2['valence'].mean()}")
    print("\n")
    print(f"Average Tempo in {year1}: {songs_y1['tempo'].mean()}")
    print(f"Average Tempo in {year2}: {songs_y2['tempo'].mean()}")
    
    # Prepare data for plotting
    attributes = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                  'instrumentalness', 'valence', 'tempo']
    
    averages = {
        'Attribute': [],
        f'{year1}': [],
        f'{year2}': []
    }

    for attribute in attributes:
        averages['Attribute'].append(attribute)
        averages[f'{year1}'].append(songs_y1[attribute].mean())
        averages[f'{year2}'].append(songs_y2[attribute].mean())

    df = pd.DataFrame(averages)

    # Plot the bar plots
    ax = df.set_index('Attribute').plot(kind='bar', width=0.8, color=['#1f77b4', '#ff7f0e'])
    ax.figure.set_size_inches(16, 8)
    plt.title(f'Comparison of Attributes Between {year1} and {year2}')
    plt.ylabel('Average Value')
    plt.xlabel('Attributes')
    plt.xticks(rotation=45, ha='right')

    # Add values on top of each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    xytext=(0, 15),  # 10 points vertical offset
                    textcoords='offset points', 
                    ha='center', 
                    va='bottom', 
                    fontsize=15, color='black')

    plt.tight_layout()
    plt.savefig('charts/year_compare.png')
    
def compare_number_of_weeks(sep):
    # Load the dataset
    data = pd.read_csv('dataset.csv')
    data = data[data['Number of Weeks On Top'] > 0]


    songs_sep_high = data[data['Number of Weeks On Top'] >= sep]
    songs_sep_low = data[data['Number of Weeks On Top'] < sep]
    
    print(f"Songs in HIGH: {len(songs_sep_high)}")
    print(f"Songs in LOW: {len(songs_sep_low)}")
    print("\n")
    print(f"Average Danceability in HIGH: {songs_sep_high['danceability'].mean()}")
    print(f"Average Danceability in LOW: {songs_sep_low['danceability'].mean()}")
    print("\n")
    print(f"Average Energy in HIGH: {songs_sep_high['energy'].mean()}")
    print(f"Average Energy in LOW: {songs_sep_low['energy'].mean()}")
    print("\n")
    print(f"Average Loudness in HIGH: {songs_sep_high['loudness'].mean()}")
    print(f"Average Loudness in LOW: {songs_sep_low['loudness'].mean()}")
    print("\n")
    print(f"Average Speechiness in HIGH: {songs_sep_high['speechiness'].mean()}")
    print(f"Average Speechiness in LOW: {songs_sep_low['speechiness'].mean()}")
    print("\n")
    print(f"Average Acousticness in HIGH: {songs_sep_high['acousticness'].mean()}")
    print(f"Average Acousticness in LOW: {songs_sep_low['acousticness'].mean()}")
    print("\n")
    print(f"Average Instrumentalness in HIGH: {songs_sep_high['instrumentalness'].mean()}")
    print(f"Average Instrumentalness in LOW: {songs_sep_low['instrumentalness'].mean()}")
    print("\n")
    print(f"Average Valence in HIGH: {songs_sep_high['valence'].mean()}")
    print(f"Average Valence in LOW: {songs_sep_low['valence'].mean()}")
    print("\n")
    print(f"Average Tempo in HIGH: {songs_sep_high['tempo'].mean()}")
    print(f"Average Tempo in LOW: {songs_sep_low['tempo'].mean()}")
    print("\n")
    print(f"Average Year in HIGH: {songs_sep_high['Year'].mean()}")
    print(f"Average Year in LOW: {songs_sep_low['Year'].mean()}")
    # Prepare data for plotting
    attributes = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                  'instrumentalness', 'valence', 'tempo', 'Year']
    
    averages = {
        'Attribute': [],
        'HIGH': [],
        'LOW': []
    }

    for attribute in attributes:
        averages['Attribute'].append(attribute)
        averages['HIGH'].append(songs_sep_high[attribute].mean())
        averages['LOW'].append(songs_sep_low[attribute].mean())
        
    df = pd.DataFrame(averages)

    # Plot the bar plots
    ax = df.set_index('Attribute').plot(kind='bar', width=1, color=['#2ca02c', '#d62728'])
    ax.figure.set_size_inches(16, 8)
    plt.title(f'Comparison of Attributes for Songs with High and Low Number of Weeks on Top')
    plt.ylabel('Average Value')
    plt.xlabel('Attributes')
    plt.xticks(rotation=45, ha='right')

    # Add values on top of each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    xytext=(0, 10),
                    textcoords='offset points', 
                    ha='center', 
                    va='bottom', 
                    fontsize=14, color='black')

    plt.tight_layout()
    plt.savefig('charts/week_on_top_compare.png')
    
    
plt.rcParams.update({'font.size': 15})
compare_hits_year(2006, 2018)

compare_number_of_weeks(40)