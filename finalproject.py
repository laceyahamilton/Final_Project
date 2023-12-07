import streamlit as st 
import plotly.figure_factory as ff
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os 
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

st.title("Song Recommendation System")
df = pd.read_csv("spotify_songs.csv")
st.image("playlists.jpg")
st.subheader("Using playlists with your favorite song to recommend new songs to you.")
 
 #tabs
intro_tab, dataset_tab, artist_tab, classification_tab, recommendation_tab, conclusion_tab = st.tabs(["Introduction", "About the Dataset", "Song Popularity", "Classification Methods", "Spotify API Recommendation", "Conclusion"])

with intro_tab:
    st.write("Once you find a song you like, you immediately replay it hundreds of times and immediately add it to a playlist with songs that feel the same. People make songs for different moods like sadness, happiness, or anger. Or they make playlists for activities like studying or driving to work. But to make those playlists you have to keep finding songs that match the purpose of the playlist. So for whatever application that you listen to music with, they should have a good recommendation system to give you more songs that you may enjoy. Here we are looking at Spotify playlists and using Spotify API to recommend new songs based on genres, artists and songs that you like or one that you would to get into.")
    st.image("musicmain.jpg")

genre_dis = px.histogram(df, x='playlist_genre', color='playlist_genre',text_auto = True)
genre_dis.update_layout(showlegend=False)

subgenre_dis = px.histogram(df, x='playlist_subgenre', color='playlist_subgenre', text_auto = True)
subgenre_dis.update_layout(showlegend=False)

df[['playlist_genre', 'playlist_subgenre']] = df[['playlist_genre', 'playlist_subgenre']] \
                                                  .apply(lambda x: x.str.capitalize(), axis=1)
genre_sub_pie = px.sunburst(df,
                  path=['playlist_genre', 'playlist_subgenre'], 
                  color='track_popularity', 
                  labels={'track_popularity': 'Popularity'})


with dataset_tab:
    st.write("From here we can see that the most frequent playlist made in this Spotify dataset is EDM. This may be surprising to many but EDM has been on the rise due to its popularity at live shows and festivals as well as in bars and clubs.")
    st.plotly_chart(genre_dis)
    st.write(" Spotify recognizes that one general genre is not enough to really describe the vibe of the playlist. So there also subgenres that give more specific about the playlist. The most frequent subgenre is progressive electro house which is a subgenre of EDM. It is a style of house music that is known for its infectious melodies and gradual build-ups in tempo and sound.")
    st.plotly_chart(subgenre_dis)
    st.markdown("[For history about genres and subgenres click here](https://musicmap.info/)")

    st.write("To get a clear picture of what subgenres belong in each genre, here is a pie chart that separates them. These subgenres allow you to get really specific with your playlists aid in finding more songs that fit the subgenre. The colors here represent overall popularity. This is different from the frequency of the playlist made because pop may have fewer playlists than EDM but those playlist are getting played way more often than the EDM playlists. Just because they are being made does not mean they are being played as often")
    st.plotly_chart(genre_sub_pie)

with artist_tab:
    st.title("Song Popularity By Artist")
    st.write("This allows you to see the most popular song by each artist in the dataset. Choose an artist from the dropdown below.")
    artist_options = df['track_artist'].unique()
    selected_artist = st.selectbox("Select an Artist:", artist_options)
    track_data = df[df['track_artist'] == selected_artist].groupby('track_name')['track_popularity'].mean().sort_values(ascending=False)
    # Plotting
    plt.figure(figsize=(10, 20))
    sns.barplot(x=track_data.values, y=track_data.index, palette='magma')
    plt.title(selected_artist + " top hits sorted by the mean of their popularity")
    plt.xlabel("Popularity")
    plt.ylabel("Tracks")
    # Display the plot in Streamlit
    st.pyplot(plt)


with classification_tab:
    st.title("Classification of Genre")
    st.write("Using feature selection, danceability, speechiness, tempo showed to be the 3 most import features in determining a genre. Using those three features, Decision Tree, K Nearest Neighbors, XGBoost and Neual Network classification models were created. Use the selector to see the results of each classifier. Hyperparameter was used for each model for optimal results. One thing that is common with each model is that EDM, Rap and Rock genres consistently do well. The models seem to have differentiating between Pop, Latin and R&B.")
    choose = st.radio(
            "Choose a Classifier",
            ["Decision Tree Classifier", "K Nearest Neighbors Classifier",
            "XGBoost Classifier", "Neural Network"])
    if choose == "Decision Tree Classifier":
        st.title("Decision Tree")
        st.write("It is a ML algorithm that recursively splits the dataset into subsets based on the most important feature at each node of the tree. Due to its hierarchical tree structure, each branch represent the outcome of that decision and each leaf node represents a final prediction. It can handle categorical and numerical features.")

        features = ['danceability', 'speechiness', 'tempo']
        target_column = 'playlist_genre'

        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])

        # Creating feature set and target variable
        X = df[features]
        y = df[target_column]

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a RandomizedSearchCV instance
        decision_tree = DecisionTreeClassifier(max_depth = 10, min_samples_split = 15)

        # Fit the random search to the data
        decision_tree.fit(X_train, y_train)

        # Use the best estimator for prediction and evaluation
        y_pred = decision_tree.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)

        st.write("Accuracy with Best Estimator:", accuracy)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=df[target_column].unique())
        cm_df = pd.DataFrame(cm, index=df[target_column].unique(), columns=df[target_column].unique())

        # Classification Report
        classification_rep = classification_report(y_test, y_pred)

        # Display the confusion matrix plot
        st.write("Confusion Matrix:")
        fig = ff.create_annotated_heatmap(z=cm_df.values, x=cm_df.columns.tolist(), y=cm_df.index.tolist(), colorscale='Viridis')
        st.plotly_chart(fig)

        # Display the classification report
        st.write("Classification Report:")
        st.text(classification_rep)

    if choose == "K Nearest Neighbors Classifier":
        st.title("K Nearest Neighbors")
        st.write("K Nearest Neighbors (KNN) is a simple ML model used for classification and regression. Its goal is predict class or value of a new data point based on the average or majority class/value of the KNN in the feature space. Its importance can depend on the characteristics of the dataset and parameters.")

        features = ['danceability', 'speechiness', 'tempo']
        target_column = 'playlist_genre'

        X = df[features]
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)


        knn_classifier = KNeighborsClassifier(n_neighbors=10, weights='distance', p=1)  # p=1 corresponds to Manhattan distance
        knn_classifier.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        y_predkn = knn_classifier.predict(X_test_scaled)

        # Evaluate the performance of the classifier
        accuracykn = accuracy_score(y_test, y_predkn)

        st.write("Accuracy:", accuracykn)

        # Confusion Matrix
        cmkn = confusion_matrix(y_test, y_predkn, labels=df[target_column].unique())
        cmkn_df = pd.DataFrame(cmkn, index=df[target_column].unique(), columns=df[target_column].unique())

        # Classification Report
        classification_repkn = classification_report(y_test, y_predkn)

        # Display the confusion matrix plot
        st.write("Confusion Matrix:")
        figkn = ff.create_annotated_heatmap(z=cmkn_df.values, x=cmkn_df.columns.tolist(), y=cmkn_df.index.tolist(), colorscale='Viridis')
        st.plotly_chart(figkn)

        # Display the classification report
        st.write("Classification Report:")
        st.text(classification_repkn)
    if choose == "XGBoost Classifier":
        st.title("XGBoost")
        st.write("XGBoost stands for eXtreme Gradient Boosting and is a ML algorithm for classification and regression. It is a ensemble learning method and combines the prediction of multiple weaker learner to create a strong model. It is made to handle complex relationships in the data and provide feature importance scores.")
        features = ['danceability', 'speechiness', 'tempo']
        target_column = 'playlist_genre'

        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])

        # Creating feature set and target variable
        X = df[features]
        y = df[target_column]

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)


        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        xgboost = XGBClassifier(n_estimators=200, max_depth=6, learning_rate = 0.1)

        xgboost.fit(X_train, y_train)


        # Use the best estimator for prediction and evaluation
        y_predxg =xgboost.predict(X_test)
        accuracyxg = metrics.accuracy_score(y_test, y_predxg)

        y_test_original_labels = label_encoder.inverse_transform(y_test)
        y_pred_original_labels = label_encoder.inverse_transform(y_predxg)

        st.write("Accuracy with Best Estimator:", accuracyxg)

        # Confusion Matrix
        cmxg = confusion_matrix(y_test_original_labels, y_pred_original_labels, labels=df[target_column].unique())
        cmxg_df = pd.DataFrame(cmxg, index=df[target_column].unique(), columns=df[target_column].unique())

        # Classification Report
        classificationxg_rep = classification_report(y_test_original_labels, y_pred_original_labels)
        st.write("Confusion Matrix:")
        figxg = ff.create_annotated_heatmap(z=cmxg_df.values, x=cmxg_df.columns.tolist(), y=cmxg_df.index.tolist(), colorscale='Viridis')
        st.plotly_chart(figxg)

        # Display the classification report
        st.write("Classification Report:")
        st.text(classificationxg_rep)
    if choose == "Neural Network":
        st.title("Neural Network")
        st.write("Another ML model used for solving complex problems like image recognition, natural language processing and classification tasks. Due to its powerful ability to learn intricate patterns from large amounts of data, it is applicable in a vast number of situations.")
        # Define features and target variable
        features = ['danceability', 'speechiness', 'tempo']
        target_variable = 'playlist_genre'

        # Extract features and target variable
        X = df[features]
        y = df[target_variable]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the features (optional but recommended for neural networks)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create and train the MLP classifier
        classifier = MLPClassifier(hidden_layer_sizes=(150,), max_iter=300, alpha = 0.0001, random_state=42)
        classifier.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        y_pred = classifier.predict(X_test_scaled)

        # Evaluate the performance of the classifier
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Accuracy:" , accuracy)
        st.write("Confusion Matrix:")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=np.unique(y), columns=np.unique(y))

        # Classification Report
        classification_rep = classification_report(y_test, y_pred, target_names=np.unique(y))

        fig = ff.create_annotated_heatmap(z=cm_df.values, x=cm_df.columns.tolist(), y=cm_df.index.tolist(), colorscale='Viridis')
        st.plotly_chart(fig)
        st.write("Classification Report:")
        st.text(classificationxg_rep)

with recommendation_tab:
    st.title('Spotify Track Recommender')
    st.write("Using Spotify's API, here a recommendation system based on your desired input. You have the option of getting a recommended song based on a Song+Artist input, getting the top 5 songs based on a selected genre, and getting similar artists based on a select artist. This is coming from Spotify so if you're favorite artist is in Spotify, you should be able to search it. Test it out bel")

    load_dotenv()
    sp = spotipy.Spotify(client_credentials_manager=spotipy.oauth2.SpotifyClientCredentials(
    client_id=os.environ["SPOTIPY_CLIENT_ID"],
    client_secret=os.environ["SPOTIPY_CLIENT_SECRET"]
))

    def recommend_songs_by_track(song_name, artist_name):
        results = sp.search(q=f'track:"{song_name}" artist:"{artist_name}"', type='track', limit=1)
        tracks = results['tracks']['items']

        if not tracks:
            st.write(f"No information found for the song '{song_name}' by '{artist_name}'.")
            return

        track = tracks[0]
        st.subheader(f"Recommended Songs based on '{song_name}' by '{artist_name}':")
        for recommendation in sp.recommendations(seed_tracks=[track['id']], limit=5)['tracks']:
            st.write(f"{recommendation['name']} by {', '.join(artist['name'] for artist in recommendation['artists'])}")
            st.image(recommendation['album']['images'][0]['url'], caption='Album Cover', use_column_width=True)

    def get_top_tracks_by_genre(genre):
        results = sp.search(q=f'genre:"{genre}"', type='track', limit=5)
        tracks = results['tracks']['items']

        if not tracks:
            st.write(f"No tracks found for the genre '{genre}'.")
            return

        st.subheader(f"Top 5 Tracks in the Genre '{genre}':")
        for track in tracks:
            st.write(f"{track['name']} by {', '.join(artist['name'] for artist in track['artists'])}")
            st.image(track['album']['images'][0]['url'], caption='Album Cover', use_column_width=True)
    def get_artist_info(artist_name):
        results = sp.search(q=artist_name, type='artist')
        artists = results['artists']['items']

        if not artists:
            st.write("No artist found.")
            return None

        artist = artists[0]
        st.write(f"Name: {artist['name']}")
        st.write(f"Genres: {', '.join(artist['genres'])}")
        st.write(f"Followers: {artist['followers']['total']}")

        if artist['images']:
            image_url = artist['images'][0]['url']
            st.image(image_url, caption=f"{artist['name']} Image", use_column_width=False)

    def get_artist_recommendations(selected_genres, limit=5):
        recommended_artists = []
        
        for genre_to_recommend in selected_genres:
            recommendations = sp.recommendations(seed_genres=[genre_to_recommend], limit=limit)['tracks']
            recommended_artists.extend(artist['artists'][0] for artist in recommendations)

        return recommended_artists

    # Function to get genres for a given artist
    def get_artist_genres(artist_name):
        artist_info = sp.search(q=f'artist:{artist_name}', type='artist')
        if artist_info['artists']['items']:
            return artist_info['artists']['items'][0].get('genres', [])
        else:
            return []


    choose1 = st.radio(
                "Choose a way to search Spotify.",
                ["Search by Artist", "Search by Song-Artist Name",
                "Search by Genre"])

    if choose1 == "Search by Artist":
        st.title('Artist Recommendations Based on Genres')
        artist_name_input = st.text_input('Enter an artist name:')

        if artist_name_input:
            artist_genres = get_artist_genres(artist_name_input)
            st.write(f'Genres for {artist_name_input}: {", ".join(artist_genres)}')

            artist_genres = [genre for genre in artist_genres if genre in sp.recommendation_genre_seeds()['genres']]

            selected_genres = st.multiselect('Select genres:', sp.recommendation_genre_seeds()['genres'], default=artist_genres)

            # Button to trigger recommendations
            if st.button('Get Recommendations'):
                # Get recommendations for selected genres
                recommended_artists = get_artist_recommendations(selected_genres)

                # Display recommendations
                if recommended_artists:
                    st.subheader('Recommended Artists:')
                    for artist in recommended_artists:
                        artist_name = artist['name']
                        artist_genres = ', '.join(artist.get('genres', []))
                        st.write(f"{artist_name}")
                else:
                    st.write('No recommendations found.')

    if choose1 == "Search by Song-Artist Name":
        song_name_input = st.text_input('Enter the name of the song:')
        artist_name_input_for_song = st.text_input('Enter the name of the artist for the song:')
        if song_name_input and artist_name_input_for_song:
            recommend_songs_by_track(song_name_input, artist_name_input_for_song)

    if choose1 == "Search by Genre":
        all_genres = sp.recommendation_genre_seeds()['genres']
        selected_genre = st.selectbox('Select a genre:', all_genres)
        if selected_genre:
            get_top_tracks_by_genre(selected_genre)


with conclusion_tab: 
 st.title("Conclusion")
 st.write("Based off of the results of the classifiers, we can see that distinguishing certain music genres are not all that easy. With XGBoost Classifier, the highest accracy acchieved was 45%. When the lines begin to blur, music genre starts become influenced by the individual's own taste and that is harder to detect. Using Spotify's API, you are able to input your favorite song, artist or browse through Spotify's genre's to find songs. Maybe with this app you can find some holiday songs that songs you have not heard before for the break. Happy listening!")
 st.write("![Your Awsome GIF](https://giphy.com/gifs/headphones-spongebob-squarepants-tqfS3mgQU28ko)")
 st.write("My name is Lacey Hamilton and I am a graduate student in the M.S. of Data Science Program. One of my major hobbies is building legos. I often find that I need to be doing something physical as well as mental stimulation to stay sane so building legos keeps my hands occupied. The coding of this program and my time getting the neuroscience undergraduate degree was enough mental stimulation for a long time. I figure if decided not to go to college I would've been doing the opposite of what I am doing now: being like a carpenter and then doing a mentally stimulating activity for a hobby. Not to say legos aren't mentally stimulating but it is different from wondering why you keep getting errors with your code for 6 hours. Best of both worlds. Have a great holiday season and I will be busy building my Christmas lego set after this semester is over next week and getting a new puppy. :) ")
 st.image("lego.png")



















