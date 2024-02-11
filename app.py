import wave
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import os
import pyaudio
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from skimage.transform import resize

app = Flask(__name__)

output_directory = 'newUsers'
os.makedirs(output_directory, exist_ok=True)

# Specify the complete path for the wave file
WAVE_OUTPUT_FILENAME = os.path.join(output_directory, 'datanew.wav')

# Load the pre-trained model from TensorFlow Hub
model_url = "https://tfhub.dev/google/yamnet/1"
model = hub.load(model_url)

# Load user watch history
uploaded_file_path = 'cleaned_user_watch_history.csv'
uploaded_df = pd.read_csv(uploaded_file_path)

# Load movie dataset
movie_dataset = pd.read_csv('cleaned_filtered_movie_data.csv')

# Define emotions mapping
emotions = {
    "happy": "Neutral",
    "sad": "Excitement",
    "neutral": "Happiness",
    "disgust": "Love",
    "fear": "Excitement",
    "angry": "Happiness"
}

# Assuming you have already loaded the saved model using tf.saved_model.load
loaded_model = tf.saved_model.load('saved_model/1/')

# Select a random row based on emotion and user ID
def select_random_by_emotion_and_user(new_emotion, user_id, dataframe):
    filtered_df = dataframe[(dataframe['Emotion'] == new_emotion) & (dataframe['User_ID'] == user_id)]

    if filtered_df.empty:
        return None

    random_row = filtered_df.sample(n=1).iloc[0]
    row_vector = random_row.drop(labels=['Movie Title', 'User_ID']).tolist()

    return row_vector

# Record audio from user
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 10

p = pyaudio.PyAudio()
frames = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global frames
    frames = []
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("* recording started")  # Add this line for debugging
    return jsonify({"result": "success"})


@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global frames
    p.terminate()  # Terminate PyAudio instance
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("* done recording")
    return jsonify({"result": "success"})


def extract_audio_features(audio_file):
    from pydub import AudioSegment

    audio = AudioSegment.from_wav(audio_file)
    y = np.array(audio.get_array_of_samples())
    sr = audio.frame_rate
    audio_embedding = model(tf.reshape(y, [1, -1]))['audio_embedding'].numpy()
    return audio_embedding

def detect_emotion(audio_file):
    audio_embedding = extract_audio_features(audio_file)

    
    positivity_score = np.mean(audio_embedding)
    if positivity_score > 0.5:
        return 'happy'
    else:
        return 'sad'

def get_suggested_movies(emotion):
    # Assuming you have user watch history available as 'uploaded_df'
    uploaded_file_path = 'cleaned_user_watch_history.csv'
    uploaded_df = pd.read_csv(uploaded_file_path)

    # Select a random row based on emotion and user ID
    random_row_vector = select_random_by_emotion_and_user(emotion, 1, uploaded_df)

    # Recommend movies based on numerical similarity
    if random_row_vector is not None:
        top_10_recommended_movies = recommend_movies_with_numerical_similarity(movie_dataset, random_row_vector)
    else:
        top_10_recommended_movies = []

    return top_10_recommended_movies

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        filename = 'received_audio.wav'
        save_path = os.path.join(os.getcwd(), filename)
        file.save(save_path)

        try:
            emotion = detect_emotion(save_path)
            print("Detected Emotion:", emotion)
            # Implement movie suggestion logic based on the detected emotion and return it in the response
            suggested_movies = get_suggested_movies(emotion)
            return jsonify({"result": "success", "emotion": emotion, "suggested_movies": suggested_movies})
        except Exception as e:
            print("Error:", str(e))
            return jsonify({"error": str(e)})

    return jsonify({"error": "File processing failed"})

# Recommendation function combining cosine similarity and numerical similarity
def recommend_movies_with_numerical_similarity(dataset, input_vector):
    text_vector, numerical_vector = input_vector[:-2], input_vector[-2:]
    dataset['combined_text'] = dataset[['genres', 'directors', 'actors', 'emotion']].apply(
        lambda x: ' '.join(x.astype(str)), axis=1)

    input_text_str = ' '.join(text_vector)
    appended_dataset = dataset.append({'combined_text': input_text_str}, ignore_index=True)

    vectorizer = CountVectorizer(max_features=10000)
    text_vectors = vectorizer.fit_transform(appended_dataset['combined_text'])

    text_csim = cosine_similarity(text_vectors[-1], text_vectors[:-1])

    dataset['norm_rating'] = (dataset['tomatometer_rating'] - dataset['tomatometer_rating'].min()) / (
            dataset['tomatometer_rating'].max() - dataset['tomatometer_rating'].min())
    dataset['norm_count'] = (dataset['tomatometer_count'] - dataset['tomatometer_count'].min()) / (
            dataset['tomatometer_count'].max() - dataset['tomatometer_count'].min())

    numerical_vector_normalized = [
        (numerical_vector[0] - dataset['tomatometer_rating'].min()) / (
                dataset['tomatometer_rating'].max() - dataset['tomatometer_rating'].min()),
        (numerical_vector[1] - dataset['tomatometer_count'].min()) / (
                dataset['tomatometer_count'].max() - dataset['tomatometer_count'].min())
    ]
    numerical_distance = np.sqrt(
        (dataset['norm_rating'] - numerical_vector_normalized[0]) ** 2 +
        (dataset['norm_count'] - numerical_vector_normalized[1]) ** 2)
    numerical_similarity = 1 / (1 + numerical_distance)

    combined_similarity = (text_csim.flatten() + numerical_similarity) / 2

    top_10_indices = np.argsort(combined_similarity)[::-1][:10]
    top_10_movies = dataset.iloc[top_10_indices]['movie_title'].tolist()

    return top_10_movies

# Defining and evaluating classifiers
# Load movie dataset with emotions
data = pd.read_csv("movie_dataset_emotions.csv")

# Preprocessing
data['genres'] = data['genres'].str.split(', ')
mlb = MultiLabelBinarizer()
genres_encoded = pd.DataFrame(mlb.fit_transform(data['genres']), columns=mlb.classes_)

label_encoder = LabelEncoder()
data["emotion"] = label_encoder.fit_transform(data["emotion"])

data = pd.concat([data['emotion'], genres_encoded], axis=1)

X = data.drop(['emotion'], axis=1)
y = data['emotion']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classifiers
classifiers = [
    ("Random Forest", RandomForestClassifier(random_state=42)),
    ("SVM", SVC(random_state=42)),
    ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
    ("Naive Bayes", GaussianNB()),
    ("Gradient Boosting", GradientBoostingClassifier()),
    ("Decision Tree Classifier", DecisionTreeClassifier()),
]

# Perform cross-validation for each classifier
for clf_name, clf in classifiers:
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"{clf_name} - Accuracy: {scores.mean():.4f} (std: {scores.std():.4f})")

if __name__ == '__main__':
    app.run(debug=True)
