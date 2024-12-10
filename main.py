import time
from collections import defaultdict
import json
import requests
from datetime import datetime, timedelta
import spotipy
from requests import ReadTimeout
from spotipy import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials
import os
import re

found_with_main = 0
found_with_secondary = 0
not_found = 0


def get_valid_dates():
    file_path = "./valid_dates.json"
    with open(file_path, "r") as file:
        valid_dates = json.load(file)
        return list(map(lambda x: datetime.strptime(x, "%Y-%m-%d"), valid_dates))


def save_results(data, output_path):
    with open(output_path, "w") as outfile:
        json.dump(data, outfile, indent=4)


def load_data(filename):
    with open(filename, "r") as file:
        return json.load(file)


def fetch_songs_for_period(start_date_str, duration_years):
    valid_dates = get_valid_dates()
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = start_date + timedelta(days=365 * duration_years)

    # Filter valid dates within the given period
    valid_dates = [date for date in valid_dates if start_date <= date < end_date]

    # Filter valid dates within the given period
    dates_by_year = defaultdict(list)
    for date in valid_dates:
        year = date.year
        dates_by_year[year].append(date)

    yearly_top_100 = defaultdict(list)
    # Process each year's data in batches
    for year, dates in dates_by_year.items():
        song_counts = defaultdict(int)
        time.sleep(20)
        print(f"Processing year: {year}")
        for date in dates:
            # Fetch data from the API
            url = f"https://raw.githubusercontent.com/mhollingshead/billboard-hot-100/main/date/{date.strftime('%Y-%m-%d')}.json"
            response = requests.get(url)
            response.raise_for_status()
            # Parse the chart data
            chart_data = response.json()
            for entry in chart_data["data"]:
                key = (entry['song'], entry['artist'])  # Tuple (music name, artist)
                song_counts[key] += 1

        # Process and keep top 100 songs for the year
        yearly_top_100[year] = sorted(song_counts.items(), key=lambda x: x[1], reverse=True)[:100]

    return yearly_top_100


def initialize_spotify_client(client_id, client_secret):
    return spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    ))


def get_spotify_id(spotify_client, song_name, artist_name):
    global found_with_main, found_with_secondary, not_found
    """
    Fetches the Spotify ID for a specific song by a given artist.

    Parameters:
    - song_name (str): The name of the song.
    - artist_name (str): The name of the artist.

    Returns:
    - dict: A dictionary containing the song name, artist name, and Spotify ID.
    - None: If no match is found.
    """
    # Construct the search query
    query = f"track:{song_name} artist:{artist_name}"

    # Perform the search
    results = spotify_call(spotify_client.search, q=query, type="track", limit=1)

    # Extract the first result if available
    if results["tracks"]["items"]:
        found_with_main += 1
        track = results["tracks"]["items"][0]
        return track["id"]
    else:
        query = f"track:{song_name} artist:{artist_name.split()[0]}"
        results = spotify_call(spotify_client.search, q=query, type="track", limit=1)
        if results["tracks"]["items"]:
            found_with_secondary += 1
            track = results["tracks"]["items"][0]
            return track["id"]
        else:
            not_found += 1
            print(f"No id found, even with relaxed criteria. Song: {song_name}, Artist: {artist_name}")


def spotify_call(api_call, *args, **kwargs):
    """
    Encapsulates a call to the Spotify API with retry logic.

    Args:
        api_call (callable): A function or method that makes the Spotify API call.
        *args: Positional arguments to pass to the api_call.
        **kwargs: Keyword arguments to pass to the api_call.

    Returns:
        The result of the API call if successful.

    Raises:
        SpotifyException: If the API call fails after 3 retries.
    """
    max_retries = 3
    retry_delay = 45

    for attempt in range(max_retries):
        try:
            return api_call(*args, **kwargs)
        except (SpotifyException, ReadTimeout) as e:
            if attempt < max_retries - 1:
                print(f"API call failed: {e}. Retrying in {retry_delay} seconds... (Attempt {attempt + 1})")
                time.sleep(retry_delay)
            else:
                print("API call failed after 3 attempts. Raising exception.")
                raise

def get_audio_features(spotify_client, spotify_ids):
    """
    Fetch the audio features for a list of Spotify track IDs.

    :param spotify_client: Initialized Spotify API client
    :param spotify_ids: List of Spotify track IDs
    :return: Dictionary of Spotify track IDs to their audio features
    """
    step = 1
    audio_features = {}

    # Process IDs in batches of 100
    for i in range(0, len(spotify_ids), step):
        batch_ids = spotify_ids[i:i + step]  # Get the next batch of up to 100 IDs
        # Fetch audio features for the current batch
        features = spotify_call(spotify_client.audio_features, batch_ids)
        for feature in features:
            if feature is not None:
                audio_features[feature['id']] = feature

    return audio_features


import pandas as pd


def create_dataframe(yearly_top_100, spotify_ids):
    """
    Merges yearly_top_100 dictionary and spotify_ids dictionary into a Pandas DataFrame.

    Args:
        yearly_top_100 (dict): Dictionary containing top 100 songs data for each year.
                               Format: {year: [[[song_name, artist], num_days], ...]}
        spotify_ids (dict): Dictionary containing Spotify IDs mapped to song and artist.
                            Format: {spotify_id: [song_name, artist]}

    Returns:
        pd.DataFrame: DataFrame containing the merged data with columns:
                      ['Year', 'Song Name', 'Artist', 'Number of Days', 'Spotify ID'].
    """
    data = []
    for year, songs in yearly_top_100.items():
        for song_info in songs:
            song_name, artist = song_info[0]
            num_weeks_on_top = song_info[1]

            spotify_id = None
            for sid, details in spotify_ids.items():
                if details[0] == song_name and details[1] == artist:
                    spotify_id = sid
                    break

            data.append([year, song_name, artist, num_weeks_on_top, spotify_id])

    df = pd.DataFrame(data, columns=['Year', 'Song Name', 'Artist', 'Number of Weeks On Top', 'Spotify ID'])
    return df

def count_missing_values(df):
    """
    Counts the number of missing values in each column of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to check.

    Returns:
        pd.Series: A Series where the index is the column name and the value is the count of missing values.
    """
    missing_counts = df.isnull().sum()
    # Filter to show only columns with missing values
    missing_columns = missing_counts[missing_counts > 0]
    return missing_columns


def preprocess_text(text):
    """
    Preprocess the text: lowercase, remove special characters, and tokenize into words.

    Args:
        text (str): The text to preprocess.

    Returns:
        set: A set of lowercase tokens (words).
    """
    # Lowercase the text and remove non-alphanumeric characters (including punctuation)
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # Tokenize by splitting on spaces and remove stopwords
    words = set(text.split())
    return words


def match_dataframes(df1, df2):
    matched_rows = []

    df1['song_name_tokens'] = df1['Song Name'].apply(preprocess_text)
    df1['artist_tokens'] = df1['Artist'].apply(preprocess_text)
    df2['track_name_tokens'] = df2['track_name'].apply(preprocess_text)
    df2['artist_tokens'] = df2['track_artist'].apply(preprocess_text)

    df1 = df1[(df1['song_name_tokens'].apply(len) > 0) & (df1['artist_tokens'].apply(len) > 0)]
    df2 = df2[(df2['track_name_tokens'].apply(len) > 0) & (df2['artist_tokens'].apply(len) > 0)]

    matches = 0
    # Compare each row in df1 with each row in df2
    for i, row1 in df1.iterrows():
        for j, row2 in df2.iterrows():
            # Check if song name matches
            song_name_match = row1['song_name_tokens'].issubset(row2['track_name_tokens']) or row2[
                'track_name_tokens'].issubset(row1['song_name_tokens'])
            # Check if artist name matches
            artist_match = row1['artist_tokens'].issubset(row2['artist_tokens']) or row2['artist_tokens'].issubset(
                row1['artist_tokens'])

            # If both conditions are satisfied, it's a match
            if song_name_match and artist_match:
                matches += 1
                matched_rows.append({
                    'df1_index': i,
                    'df2_index': j,
                    'Song Name (df1)': row1['Song Name'],
                    'Artist (df1)': row1['Artist'],
                    'Song Name (df2)': row2['track_name'],
                    'Artist (df2)': row2['track_artist']
                })
                break
    print(f"Matched {matches} rows, from a total of {len(df1)} rows.")
    # Create a DataFrame with the matched rows
    matched_df = pd.DataFrame(matched_rows)
    return matched_df

def main(start_date_str, duration_years):
    top_100_by_year_file = "./yearly_top_100.json"
    spotify_ids_file = "./spotify_ids.json"
    dataframe_file = "./data.csv"
    songs_dataset_file = "./datasets/spotify_songs.csv"
    spotify_client = initialize_spotify_client('76efa3b6bd924968a46336ceb7502225', 'deadf5cd6532478693b1c43631b362f5')

    if not os.path.isfile(top_100_by_year_file):
        results = fetch_songs_for_period(start_date_str, duration_years)
        save_results(results, top_100_by_year_file)
    top_100_by_year = load_data(top_100_by_year_file)

    if not os.path.isfile(spotify_ids_file):
        spotify_ids_per_song = dict()
        for songs in top_100_by_year.values():
            for (song, artist), _ in songs:
                spotify_id = get_spotify_id(spotify_client, song, artist)
                spotify_ids_per_song[str(spotify_id)] = (song, artist)
        print(f"Found {found_with_main} songs at first try, {found_with_secondary} songs at second try and did not found {not_found} songs.\nTotal songs found: {found_with_main + found_with_secondary}.")
        save_results(spotify_ids_per_song, spotify_ids_file)
    spotify_ids_per_song = load_data(spotify_ids_file)

    if not os.path.isfile(dataframe_file):
        df = create_dataframe(top_100_by_year, spotify_ids_per_song)
        df.to_csv(dataframe_file, index=False)

    df1 = pd.read_csv(dataframe_file)
    df2 = pd.read_csv(songs_dataset_file)
    df = match_dataframes(df1, df2)
    df.to_csv('all.csv', index=False)
    print()




if __name__ == "__main__":
    main("2000-01-01", 24)
