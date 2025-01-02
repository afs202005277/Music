import time
from collections import defaultdict
import json

import numpy as np
import requests
from datetime import datetime, timedelta
import spotipy
from requests import ReadTimeout
from spotipy import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials
import os
import re
import multiprocessing as mp
import pandas as pd

found_with_main = 0
found_with_secondary = 0
not_found = 0


def process_csv(file_name):
    """
    Processes a CSV file to count the number of rows associated with each unique 'Year'
    and prints the results. This function checks for the existence of a 'Year' column in
    the CSV file, and if present, it calculates and displays the count of rows grouped
    by the values in the 'Year' column. In the absence of the file, or if other issues
    occur, appropriate messages or errors are printed.

    :param file_name: The path to the CSV file to be processed
    :type file_name: str
    :returns: None
    :rtype: NoneType
    """
    try:
        df = pd.read_csv(file_name)

        if 'Year' not in df.columns:
            print("The 'Year' column is not found in the CSV file.")
            return

        # df = df.dropna(subset=['Year'])

        counts_per_year = df['Year'].value_counts().sort_index()

        print("Number of rows per year:")
        print(counts_per_year)

    except FileNotFoundError:
        print(f"The file '{file_name}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


def get_valid_dates():
    file_path = "./datasets_caches/valid_dates.json"
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
    """
    Fetches the top 100 songs for a given period by utilizing the Billboard Hot 100 charts data.

    The function retrieves data for valid dates that fall within the specified start date
    and duration in years. It processes these dates to group them by year and fetches
    Billboard Hot 100 chart data via API calls. After processing the data, the most
    frequent 100 songs for each year within the period are identified and returned.

    :param start_date_str: A string representing the start date in the "YYYY-MM-DD" format.
    :param duration_years: An integer representing the duration in years from the start date.
    :return: A dictionary where keys are years and values are lists of the top 100 songs
             for those years, sorted by frequency of occurrence.
    :rtype: defaultdict[int, list[tuple[str, str]]]
    """
    valid_dates = get_valid_dates()
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = start_date + timedelta(days=365 * duration_years)

    valid_dates = [date for date in valid_dates if start_date <= date < end_date]

    dates_by_year = defaultdict(list)
    for date in valid_dates:
        year = date.year
        dates_by_year[year].append(date)

    yearly_top_100 = defaultdict(list)
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

        yearly_top_100[year] = sorted(song_counts.items(), key=lambda x: x[1], reverse=True)[:100]

    return yearly_top_100


def initialize_spotify_client(client_id, client_secret):
    """
    Initializes a Spotify client using client credentials.

    This function sets up a Spotify client by using the provided client ID and
    client secret to authenticate via the SpotifyClientCredentials manager.

    :param client_id: A string representing the Spotify application's client ID.
    :param client_secret: A string representing the Spotify application's client secret.
    :return: An instance of the Spotify client initialized using the given credentials.
    """
    return spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    ))


def get_spotify_id(spotify_client, song_name, artist_name):
    """
    Searches for a Spotify track ID based on the provided song name and artist name. Initially, it
    uses the full artist name to perform the query. If no result is found, a secondary search
    with only the first part of the artist name is executed. Tracks the number of successful
    searches with primary and secondary criteria, as well as searches that do not yield a result.

    :param spotify_client: Spotify client object used to interact with the Spotify API.
    :param song_name: The name of the song to look up.
    :type song_name: str
    :param artist_name: The name of the artist associated with the song.
    :type artist_name: str
    :return: The Spotify ID of the track if found, otherwise None.
    :rtype: str or None
    """
    global found_with_main, found_with_secondary, not_found
    query = f"track:{song_name} artist:{artist_name}"

    results = spotify_call(spotify_client.search, q=query, type="track", limit=1)

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
    Executes a given Spotify API call with retry logic. This function attempts to
    execute the provided API call up to a maximum number of retries. If the API
    call fails due to specific exceptions, it waits for a specified delay before
    retrying. On the final failed attempt, the exception is raised.

    :param api_call: A callable representing the Spotify API operation to be executed.
    :param args: Positional arguments to pass to the provided `api_call`.
    :param kwargs: Keyword arguments to pass to the provided `api_call`.
    :return: The result of the successful `api_call` execution.
    :rtype: Any
    :raises SpotifyException: If the API call fails after all retries due to an
        issue with Spotify API operations.
    :raises ReadTimeout: If the API call fails after all retries due to a timeout
        in reading the response.
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
    DEPRECATED: this function uses an endpoint that no longer works.
    Fetches audio features for a list of Spotify track IDs using the provided Spotify client.

    This function retrieves audio features for tracks in batches and aggregates them
    into a dictionary where the track IDs are the keys and the audio features are
    the corresponding values. It skips any tracks that do not have audio features
    available.

    :param spotify_client: A Spotify client object used to fetch audio features.
    :param spotify_ids: A list of Spotify track IDs for which audio features
        are to be fetched.
    :return: A dictionary where the keys are Spotify track IDs and the values are
        corresponding audio feature data.
    :rtype: dict
    """
    step = 1
    audio_features = {}

    for i in range(0, len(spotify_ids), step):
        batch_ids = spotify_ids[i:i + step]
        features = spotify_call(spotify_client.audio_features, batch_ids)
        for feature in features:
            if feature is not None:
                audio_features[feature['id']] = feature

    return audio_features


def create_dataframe(yearly_top_100, spotify_ids):
    """
    Generates a pandas DataFrame from the given yearly top songs data and their corresponding Spotify ID
    mapping. It iterates through each yearly top 100 list and matches song names and artists with their
    Spotify IDs. The resulting DataFrame contains detailed song information including the year, song
    name, artist, number of weeks the song was on top, and the Spotify ID.

    :param yearly_top_100: A dictionary where keys are years (int) and values are lists of song data.
        Each list contains tuples with the first element being a tuple containing song name and artist
        (both strings), and the second element is the number of weeks the song was on top (int).
    :param spotify_ids: A dictionary mapping Spotify IDs (str) to a tuple of song name (str) and artist
        (str).
    :return: A pandas DataFrame with columns "Year", "Song Name", "Artist", "Number of Weeks On Top",
        and "Spotify ID". Each row represents a song with the respective details.
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
    missing_counts = df.isnull().sum()
    missing_columns = missing_counts[missing_counts > 0]
    return missing_columns


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)

    words = set(text.split())
    return words


def match_dataframes(df1, df2):
    """
    Match data from two dataframes by comparing tokens derived from the song and artist names.
    Tokens are obtained by preprocessing these names into sets of elements for comparison.
    Matches are identified when tokens from one dataframe are subsets of or exactly equal to
    tokens from the other dataframe for both song names and artists.

    :param df1: The first dataframe which contains the columns "Song Name" and "Artist".
    :param df2: The second dataframe which contains the columns "track_name" and "track_artist".
    :return: A tuple containing two elements: a dataframe of matched rows from both input
             dataframes, and a list of rows without a match from the first dataframe.
    """
    matched_rows = []
    no_matches = []
    df1['song_name_tokens'] = df1['Song Name'].apply(lambda x: set(preprocess_text(x)))
    df1['artist_tokens'] = df1['Artist'].apply(lambda x: set(preprocess_text(x)))
    df2['track_name_tokens'] = df2['track_name'].apply(lambda x: set(preprocess_text(x)))
    df2['artist_tokens'] = df2['track_artist'].apply(lambda x: set(preprocess_text(x)))

    df1 = df1[(df1['song_name_tokens'].apply(len) > 0) & (df1['artist_tokens'].apply(len) > 0)]
    df2 = df2[(df2['track_name_tokens'].apply(len) > 0) & (df2['artist_tokens'].apply(len) > 0)]

    matches = 0
    for i, row1 in df1.iterrows():
        found = False
        for _, row2 in df2.iterrows():
            song_name_match = row1['song_name_tokens'].issubset(row2['track_name_tokens']) or row2[
                'track_name_tokens'].issubset(row1['song_name_tokens'])
            artist_match = row1['artist_tokens'].issubset(row2['artist_tokens']) or row2['artist_tokens'].issubset(
                row1['artist_tokens'])

            if song_name_match and artist_match:
                found = True
                matches += 1
                matched_rows.append({**row1.to_dict(), **row2.to_dict()})
                break
        if not found:
            no_matches.append([row1['Song Name'], row1['Artist']])

    print(f"Matched {matches} rows, from a total of {len(df1)} rows.")
    matched_df = pd.DataFrame(matched_rows)
    return matched_df, no_matches


def match_dataframes_worker(args):
    chunk, df2 = args
    return match_dataframes(chunk, df2)


def parallel_match_dataframes(df1, df2):
    """
    Matches rows from two dataframes in parallel using multiprocessing.

    This function splits the first dataframe into chunks based on the number of
    available CPU cores and then processes each chunk in parallel. The matching
    results are combined, and unmatched rows are saved to a JSON file. The
    function finally returns the combined dataframe of matched rows.

    :param df1: The first dataframe to be matched.
    :type df1: pandas.DataFrame
    :param df2: The second dataframe to be matched.
    :type df2: pandas.DataFrame

    :return: A dataframe containing only the matched rows from the first dataframe.
    :rtype: pandas.DataFrame
    """
    cpu_count = mp.cpu_count()

    df1_chunks = np.array_split(df1, cpu_count)

    print(f"Processing {len(df1_chunks)} chunks of {len(df1_chunks[0])} rows each.")

    worker_args = [(chunk, df2) for chunk in df1_chunks]

    with mp.Pool(cpu_count) as pool:
        results = pool.map(match_dataframes_worker, worker_args)

    matched_rows = [match for (match, _) in results]
    no_matched_rows = [no_match for (_, no_match) in results]

    save_results(no_matched_rows, './datasets_caches/no_matches.json')
    final_df = pd.concat(matched_rows, ignore_index=True)

    return final_df


def merge_spotify_datasets(datasets_with_translations):
    renamed_datasets = []
    for dataset, col_renaming in datasets_with_translations:
        renamed_datasets.append(dataset.rename(columns=col_renaming))
    return pd.concat(renamed_datasets, axis=0, join='inner', ignore_index=True)


def join_dataframes_on_spotify_id(df1, df2):
    joined_df = pd.merge(df1, df2, on='Spotify ID', how='inner')

    unmatched_df1 = df1[~df1['Spotify ID'].isin(joined_df['Spotify ID'])]

    return joined_df, unmatched_df1


def join_dataframes_vertically(df1, df2):
    common_columns = df1.columns.intersection(df2.columns)

    df1_common = df1[common_columns]
    df2_common = df2[common_columns]

    vertically_joined_df = pd.concat([df1_common, df2_common], axis=0)

    return vertically_joined_df


def get_songs_to_add(df_hits, df_general):
    """
    Filters and prepares a list of songs to be added to a playlist based on the comparison
    between a DataFrame of hit songs and another DataFrame of general songs. The function
    identifies songs present in the general data set but not featured in the hit songs list
    for each year. These songs are further ranked and filtered based on their popularity.

    :param df_hits: A pandas DataFrame containing hit songs information. It is expected to
        include columns 'Year', 'Spotify ID', 'track_artist', and 'Song Name'.
    :param df_general: A pandas DataFrame containing general songs data. It is expected to
        include columns 'year', 'Spotify ID', 'track_artist', 'track_name',
        and 'track_popularity'.
    :return: A pandas DataFrame containing the filtered and prepared list of songs to be
        added, including their modified attributes. The columns include 'Year', 'genre',
        'track_artist', 'track_name', 'track_popularity', and 'Number of Weeks On Top'.
    """
    songs_to_add = []
    for year in df_hits['Year'].unique():
        hits_year = df_hits[df_hits['Year'] == year]
        general_year = df_general[df_general['year'] == year]

        filtered_songs = general_year[~(
                general_year['Spotify ID'].isin(hits_year['Spotify ID']) |
                (
                        general_year['track_artist'].isin(hits_year['track_artist']) &
                        general_year['track_name'].isin(hits_year['Song Name'])
                )
        )]

        filtered_songs = filtered_songs.sort_values(by='track_popularity', ascending=True)
        songs_to_add.extend(filtered_songs.head(100).to_dict('records'))

    final_df = pd.DataFrame(songs_to_add)
    final_df = final_df.rename(columns={'year': 'Year', 'playlist_genre': 'genre'})
    final_df['Number of Weeks On Top'] = 0

    return final_df


def main(start_date_str, duration_years):
    """
    This function processes and merges multiple datasets related to song data. It performs various operations
    such as fetching song data for a given time period, mapping Spotify IDs, creating dataframes, unifying
    datasets, and ensuring data consistency. The final processed dataset is saved to a file for further use.
    The function creates intermediate caches to optimize operations and avoid redundant calls.

    :param start_date_str: A string representing the starting date for the data processing. It is
       used to determine the time period for fetching songs.
    :type start_date_str: str
    :param duration_years: An integer indicating the number of years from the start date for which
       data will be fetched and processed.
    :type duration_years: int
    :return: None
    """
    final_dataset_file = "./dataset.csv"
    if not os.path.isfile(final_dataset_file):
        top_100_by_year_file = "./datasets_caches/yearly_top_100.json"
        spotify_ids_file = "./datasets_caches/spotify_ids.json"
        dataframe_file = "./datasets_caches/data.csv"
        final_hits_dataset_file = "./datasets_caches/hits.csv"
        million_songs_dataset = pd.read_csv('datasets/spotify_data.csv')

        million_songs_dataset = million_songs_dataset.rename(
            columns={'genre': 'playlist_genre', 'artist_name': 'track_artist', 'popularity': 'track_popularity',
                     'track_id': 'Spotify ID'})
        spotify_client = initialize_spotify_client('76efa3b6bd924968a46336ceb7502225',
                                                   'deadf5cd6532478693b1c43631b362f5')

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
            print(
                f"Found {found_with_main} songs at first try, {found_with_secondary} songs at second try and did not found {not_found} songs.\nTotal songs found: {found_with_main + found_with_secondary}.")
            save_results(spotify_ids_per_song, spotify_ids_file)
        spotify_ids_per_song = load_data(spotify_ids_file)

        if not os.path.isfile(dataframe_file):
            df = create_dataframe(top_100_by_year, spotify_ids_per_song)
            df.to_csv(dataframe_file, index=False)

        df_2010_2019 = pd.read_csv('./datasets/2010_2019.csv')
        df_2010_2019["instrumentalness"] = np.nan
        unified_dataset = merge_spotify_datasets(
            [[pd.read_csv('./datasets/30000.csv'), {}], [pd.read_csv('./datasets/114000.csv'),
                                                         {'artists': 'track_artist',
                                                          'popularity': 'track_popularity',
                                                          'album_name': 'track_album_name',
                                                          'track_genre': 'playlist_genre'}],
             [pd.read_csv('./datasets/2000_2019.csv'),
              {'genre': 'playlist_genre', 'artist': 'track_artist', 'song': 'track_name',
               'popularity': 'track_popularity'}],
             [df_2010_2019,
              {'title': 'track_name', 'artist': 'track_artist', 'top genre': 'playlist_genre', 'nrgy': 'energy',
               'dnce': 'danceability', 'val': 'valence', 'dur': 'duration_ms', 'acous': 'acousticness',
               'spch': 'speechiness', 'pop': 'track_popularity', 'bpm': 'tempo', 'dB': 'loudness'}]])

        df1 = pd.read_csv(dataframe_file)
        matched, unmatched = join_dataframes_on_spotify_id(df1, million_songs_dataset)

        df = parallel_match_dataframes(unmatched, unified_dataset)
        df = join_dataframes_vertically(df, matched)
        df = df.drop('Artist', axis=1)
        df = df.rename(columns={'playlist_genre': 'genre'})

        cols_to_fix = ['danceability', 'energy', 'speechiness', 'acousticness', 'speechiness', 'valence']

        for col in cols_to_fix:
            df[col] = df[col].apply(lambda x: x / 100 if x > 1 else x)

        df.to_csv(final_hits_dataset_file, index=False)

        songs_to_add_non_hits = get_songs_to_add(df, million_songs_dataset)
        final = join_dataframes_vertically(df, songs_to_add_non_hits)

        final = final[~((final['track_popularity'] == 0) & (final['Number of Weeks On Top'] > 0))]

        final.to_csv(final_dataset_file, index=False)
    dataset = pd.read_csv(final_dataset_file)
    print()


if __name__ == "__main__":
    main("2000-01-01", 24)
    print(len(pd.read_csv('dataset.csv')))
    process_csv('dataset.csv')

