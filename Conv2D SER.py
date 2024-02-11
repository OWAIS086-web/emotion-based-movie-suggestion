#!/usr/bin/env python
# coding: utf-8

# # Sentiment Based Recommendations System 

# In[5]:


# Importing libraries
import os
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
from IPython.core.display import display
import seaborn as sns
from matplotlib.colors import Normalize
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
import warnings
warnings.filterwarnings("ignore")


# In[6]:


# Getting TESS data:
paths, labels, duration = [], [], []

for dirname, _, filenames in os.walk('tess'):
    for filename in filenames:
        
        paths.append(os.path.join(dirname, filename))
        
        duration.append(round(librosa.get_duration(filename=paths[-1]), 3))
        
        label = filename[::-1].split('_')[0][::-1]
        labels.append(label[:-4].lower())

df_tess = pd.DataFrame({'path':paths,'duration': duration, 'dataset': 'TESS', 'emotion':labels})

                  
df_tess.tail()


# In[ ]:





# In[7]:


# Getting RAVDESS data
paths, labels, duration = [], [], []

for dirname, _, filenames in os.walk('ravdess/'):
    for filename in filenames:
        
        paths.append(os.path.join(dirname, filename))
        
        duration.append(round(librosa.get_duration(filename=paths[-1]), 3)) 
        
        label = filename[::-1].split('_')[0][::-1]
        
        if label[6:8] == '01':
            labels.append('neutral')
        elif label[6:8] == '02':
            labels.append('calm')
        elif label[6:8] == '03':
            labels.append('happy')
        elif label[6:8] == '04':
            labels.append('sad')
        elif label[6:8] == '05':
            labels.append('angry')
        elif label[6:8] == '06':
            labels.append('fear')
        elif label[6:8] == '07':
            labels.append('disgust')
        elif label[6:8] == '08':
            labels.append('surprise')       

df_ravdess = pd.DataFrame({'path':paths,'duration': duration, 'dataset': 'RAVDESS', 'emotion':labels})

df_ravdess.sample(5)


# In[8]:


# Getting SAVEE data
paths, labels, duration = [], [], []

for dirname, _, filenames in os.walk('savee/database'):
    for filename in filenames:
        
        paths.append(os.path.join(dirname, filename))
        
        label = filename[::-1].split('_')[0][::-1]
        
        if label[:1] == 'a':
            labels.append('angry')
        elif label[:1] == 'd':
            labels.append('disgust')
        elif label[:1] == 'f':
            labels.append('fear')
        elif label[:1] == 'h':
            labels.append('happy')
        elif label[:1] == 'n':
            labels.append('neutral')
        elif label[:1] == 's':
            if label[:2] == 'sa':
                labels.append('sad')
            else:
                labels.append('surprise')

paths = paths[1:] # to filter out 'info.txt' file

for file in paths:
    duration.append(round(librosa.get_duration(filename=file), 3)) 
#print(len(paths), len(duration), len(labels))
labels = labels[:-1]
#print(len(paths), len(duration), len(labels))
df_savee = pd.DataFrame({'path':paths, 'duration': duration, 'dataset': 'SAVEE', 'emotion':labels})
                  
df_savee.sample(5)


# In[9]:


crema = 'crema/'

crema_directory_list = os.listdir(crema)

file_emotion = []
file_path = []
duration = []

for file in crema_directory_list:
    
    
    # storing file paths
    file_path.append(crema + file)
    # storing file emotions
    part=file.split('_')
    if part[2] == 'SAD':
        file_emotion.append('sad')
    elif part[2] == 'ANG':
        file_emotion.append('angry')
    elif part[2] == 'DIS':
        file_emotion.append('disgust')
    elif part[2] == 'FEA':
        file_emotion.append('fear')
    elif part[2] == 'HAP':
        file_emotion.append('happy')
    elif part[2] == 'NEU':
        file_emotion.append('neutral')
    else:
        file_emotion.append('Unknown')
        
for file in file_path:
    duration.append(round(librosa.get_duration(filename=file), 3)) 
    


df_crema = pd.DataFrame({'path':file_path, 'duration': duration, 'dataset': 'CREMA', 'emotion':file_emotion})

df_crema
# crema_df


# In[10]:


# Let's merge the datesets together, now that they have been formatted the same way:

df = pd.concat([df_tess, df_ravdess, df_savee, df_crema])

# Dropping 'calm' as out the scope (also not many samples)
df = df[df['emotion'].str.contains('calm') == False].reset_index(drop=True)
df = df[df['emotion'].str.contains('surprise') == False].reset_index(drop=True)

print('The dataset has {} audio files. Below printed 5 random entries:'.format(df.shape[0]))

df.sample(5)


# In[ ]:





# In[13]:


get_ipython().system('pip install pyaudio')
get_ipython().system('pip install wave')


# In[ ]:





# In[11]:


# Let's merge the datesets together, now that they have been formatted the same way:

df = pd.concat([df_tess, df_ravdess, df_savee, df_crema])

# Dropping 'calm' as out the scope (also not many samples)
df = df[df['emotion'].str.contains('calm') == False].reset_index(drop=True)
df = df[df['emotion'].str.contains('surprise') == False].reset_index(drop=True)

print('The dataset has {} audio files. Below printed 5 random entries:'.format(df.shape[0]))

df.sample(5)


# In[12]:


paths, labels, duration = [], [], []

for dirname, _, filenames in os.walk('newUsers'):
    for filename in filenames:
        
        paths.append(os.path.join(dirname, filename))
        
        duration.append(round(librosa.get_duration(filename=paths[-1]), 3))
        
        label = filename[::-1].split('_')[0][::-1]
        labels.append(label[:-4].lower())

df_new = pd.DataFrame({'path':paths,'duration': duration, 'dataset': 'newUsers', 'emotion':labels})

                  
df_new.head()


# In[11]:


df['emotion'].value_counts()


# In[12]:


df['path']


# # Exploratory Data Analysis (EDA)

# In[13]:


df.info()


# In[14]:


# Creating a figure with 2 subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

# Check samples distribution
df.groupby(['emotion','dataset']).size().unstack().plot(kind='bar', stacked=True, ax=axes[0])
axes[0].set_title('Distribution of audio files by target emotion and dataset', size=14)
axes[0].set_ylabel('number of samples')
axes[0].legend(title='Dataset')

# Check duration distribution by each source using violin plots
sns.violinplot(x=df['dataset'],y=df['duration'], linewidth=1, ax=axes[1])
axes[1].set_xlabel('Emotion')
axes[1].set_ylabel('Samples duration (seconds)')
axes[1].set_title('Audio samples duration distribution for each emotion', size=14)

plt.show()


# In[15]:


# function to display samples information by emotion
# note that a random samples is generated each time the function is called
# this is on purpose as to check different samples of each emotion every time

def show_audio(emotion):
    # create sublots
    fig, axs = plt.subplots(nrows=1,ncols=3, figsize=(20,4))
    # filter dataframe to emotion)
    df_show = df.loc[df['emotion'] == emotion].reset_index(drop=True)
    index = random.randint(0, df_show.shape[0])
    
    # load audio file:
    y, sr = librosa.load(df_show.path[index], sr=16000)
    
    # Show waveform
    librosa.display.waveshow(y, sr=sr, ax=axs[0])
    axs[0].set_title('Waveform')
    
    # Extract fundamental frequency (f0) using a probabilistic approach
    f0, _, _ = librosa.pyin(y, sr=sr, fmin=50, fmax=1500, frame_length=2048)

    # Establish timepoint of f0 signal
    timepoints = np.linspace(0, df_show.duration[index], num=len(f0), endpoint=False)
    
    # Compute short-time Fourier Transform
    x_stft = np.abs(librosa.stft(y))
    
    # Apply logarithmic dB-scale to spectrogram and set maximum to 0 dB
    x_stft = librosa.amplitude_to_db(x_stft, ref=np.max)
    
    # Plot STFT spectrogram
    librosa.display.specshow(x_stft, sr=sr, x_axis="time", y_axis="log", ax=axs[1])
    
    # Plot fundamental frequency (f0) in spectrogram plot
    axs[1].plot(timepoints, f0, color="cyan", linewidth=4)
    axs[1].set_title('Spectrogram with fundamental frequency')
    
    # Extract 'n_mfcc' numbers of MFCCs components - in this case 30
    x_mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)

    # Plot MFCCs
    librosa.display.specshow(x_mfccs, sr=sr, x_axis="time", norm=Normalize(vmin=-50, vmax=50), ax=axs[2])
    axs[2].set_title('MFCCs')
    
    # Show metadata in title
    plt.suptitle('File: {}  -  Emotion: {}'.format(df_show.path[index], df_show.emotion[index]), size=14)
    plt.tight_layout()
    plt.show()
    
    # Display media player for the selected file
    display(ipd.Audio(y, rate=sr))


# In[16]:


# Getting ordered list of emotions ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotions = sorted(list(df.emotion.unique()))

# Get waveforms, spectograms, mfccs and media player for each emotion
for emotion in emotions:
    show_audio(emotion)


# In[17]:


mfccs = []

for file in df.path:
    # load audio file:
    y, sr = librosa.load(file, sr=16000)
    
    # Extract 'n_mfcc' numbers of MFCCs components - in this case 30
    mfccs.append(librosa.feature.mfcc(y=y, sr=sr, fmin=50, n_mfcc=30))


# In[18]:


# Define function to resize the 2D arrays
def resize_array(array):
    new_matrix = np.zeros((30,150))   # Initialize the new matrix shape with an array 30X150 of zeros
    for i in range(30):               # Iterate rows
        for j in range(150):          # Iterate columns
            try:                                 # the mfccs of a sample will replace the matrix of zeros, then cutting the array up to 150
                new_matrix[i][j] = array[i][j]
            except IndexError:                   # if mfccs of a sample is shorter than 150, then keep looping to extend lenght to 150 with 0s
                pass
    return new_matrix

# Create a variable to store the new resized mfccs and apply function for all the extracted mfccs
resized_mfccs = []

for mfcc in mfccs:
    resized_mfccs.append(resize_array(mfcc))


# In[19]:


# Create sublots
fig, axs = plt.subplots(nrows=1,ncols=6, figsize=(20,3))

# Select 6 random MFCCs
for i in range(6):
    index = random.randint(0, len(resized_mfccs))
    
    # Plot MFCCs
    librosa.display.specshow(resized_mfccs[index], sr=sr, x_axis="time", ax=axs[i], norm=Normalize(vmin=-50, vmax=50))
    axs[i].set_title(str(df.emotion[index]) + ' - ' + str(df.duration[index]) + ' sec')

plt.suptitle('Few MFCCs of size 30x150', size=18)
plt.tight_layout()
plt.show()


# In[20]:


from sklearn.model_selection import train_test_split

# Select target
df['emotion'].replace({'angry':0,'disgust':1,'fear':2,'happy':3,'neutral':4,'sad':5}, inplace=True)
#df['emotion'].replace({0:'angry',1:'disgust',2:'fear',3:'happy',4:'neutral',5:'sad'}, inplace=True)
y = df.emotion.values

# Features
X = resized_mfccs.copy()

# Create train, validation and test set
x_tr, x_te, y_tr, y_te = train_test_split(X, y, train_size=0.9, shuffle=True, random_state=0)
x_tr, x_va, y_tr, y_va = train_test_split(x_tr, y_tr, test_size=0.3, shuffle=True, random_state=0)

# Convert data to numpy arrays
x_tr = np.array([i for i in x_tr])
x_va = np.array([i for i in x_va])
x_te = np.array([i for i in x_te])

# Plot size of data
#print(y)
print(x_tr.shape)
print(x_va.shape)
print(x_te.shape)


# In[21]:


# Get mean and standard deviation from the training set
tr_mean = np.mean(x_tr, axis=0)
tr_std = np.std(x_tr, axis=0)

# Apply data scaling
x_tr = (x_tr - tr_mean)/tr_std
x_va = (x_va - tr_mean)/tr_std
x_te = (x_te - tr_mean)/tr_std


# In[22]:


# Add the 'channel' dimension to the MFCCs spectrum input 'images'
print(f"x_tr has a dimension of {x_tr.shape} before the manipulation.")

x_tr = x_tr[..., None]
x_va = x_va[..., None]
x_te = x_te[..., None]

print(f"x_tr has a dimension of {x_tr.shape} after the manipulation.")


# In[72]:


from tensorflow import keras
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Dropout, Flatten, Dense, MaxPool2D)
from tensorflow.keras import initializers

# Create convolutional neural network and return summary
model = keras.Sequential()
model.add(Conv2D(filters=64, kernel_size=5, strides=(2, 2), activation="relu", input_shape=x_tr.shape[1:]))
model.add(MaxPool2D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=4, strides=(2, 1), activation="relu"))
model.add(MaxPool2D(pool_size=2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=7, activation="softmax"))
model.summary()


# In[75]:


# Compile the model using Adam's default learning rate
model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Create 'EarlyStopping' callback
earlystopping_cb = keras.callbacks.EarlyStopping(patience=5, verbose = 1)


# In[76]:


# Including CREMA
#%%time

#  Train the neural network
history = model.fit(
    x=x_tr,
    y=y_tr,
    epochs=80,
    batch_size=32,
    validation_data=(x_va, y_va),
    callbacks=[earlystopping_cb]
)


# In[ ]:


# # Excluding CREMA
# #%%time

# #  Train the neural network
# history = model.fit(
#     x=x_tr,
#     y=y_tr,
#     epochs=100,
#     batch_size=32,
#     validation_data=(x_va, y_va),
#     callbacks=[earlystopping_cb]
# )


# In[77]:


# Plots neural network performance metrics for train and validation
fig, axs = plt.subplots(1, 2, figsize=(20, 5))
plt.suptitle('Convolutional Netural Network with MFCCs', size=15)
results = pd.DataFrame(history.history)
results[["loss", "val_loss"]].plot(ax=axs[0])
axs[0].set_title("Validation loss {:.3f} (mean last 3)".format(np.mean(history.history["val_loss"][-3:])))
results[["accuracy", "val_accuracy"]].plot(ax=axs[1])
axs[1].set_title("Validation accuracy {:.3f} (mean last 3)".format(np.mean(history.history["val_accuracy"][-3:])))
plt.show()


# In[78]:


# Collect loss and accuracy for the test set
loss_te, accuracy_te = model.evaluate(x_te, y_te)

print("Test loss: {:.2f}".format(loss_te))
print("Test accuracy: {:.2f}%".format(100 * accuracy_te))


# In[79]:



# 


# In[32]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Compute test set predictions
predictions = model.predict(x_te)

pred = []

for i in predictions:
    pred.append(np.argmax(i))
    
from sklearn.metrics import ConfusionMatrixDisplay

labels = {'angry':0,'disgust':1,'fear':2,'happy':3,'neutral':4,'sad':5}
#labels = ['angry','disgust', 'fear', 'happy', 'neutral', 'sad']


def plot_confusion_matrices(y_true, y_pred):

    # Create two subplots
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plots the standard confusion matrix
    ax1.set_title("Confusion Matrix (counts)")
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, ax=ax1)

    # Plots the normalized confusion matrix
    ax2.set_title("Confusion Matrix (ratios)")
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, normalize="true", ax=ax2)

    plt.show()

# Plot confusion matrices
plot_confusion_matrices(y_te, pred)


# In[37]:


model.summary()


# In[80]:


from joblib import dump, load

dump(model, 'CT.joblib') 


# In[81]:


import joblib
import gzip



# Specify the filename for the compressed joblib file
compressed_filename = "model_compressed.joblib.gz"

# Create a gzip-compressed joblib file
with gzip.open(compressed_filename, "wb") as gzipped_file:
    joblib.dump(model, gzipped_file)

print(f"Model saved as {compressed_filename}")


# In[23]:


from joblib import dump, load
model_saved = load('CT.joblib') 


# In[24]:


# Collect loss and accuracy for the test set
loss_te, accuracy_te = model_saved.evaluate(x_te, y_te)

print("Test loss: {:.2f}".format(loss_te))
print("Test accuracy: {:.2f}%".format(100 * accuracy_te))


# In[26]:


def detect_emotion(audio_file):
    # Load the audio file and extract MFCCs
    y, sr = librosa.load(audio_file, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
    
    # Resize and scale the MFCCs
    resized_mfccs = resize_array(mfccs)
    scaled_mfccs = (resized_mfccs - tr_mean) / tr_std
    scaled_mfccs = np.resize(scaled_mfccs, (1, 30, 150, 1))
    
    # Predict the emotion using the trained model
    prediction = model_saved.predict(scaled_mfccs)
    emotion_index = np.argmax(prediction)
    
    # Map the emotion index back to the original labels
    emotion_mapping = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad'}
    predicted_emotion = emotion_mapping[emotion_index]
    
    return predicted_emotion


audio_file = 'newUsers\\data.wav'

#audio_file = 'Downloads/savee/database/KL_sa11.wav'
predicted_emotion = detect_emotion(audio_file)
print("Predicted emotion:", predicted_emotion)


# # Movies Recommendations System

# In[42]:


# Load the movie dataset
movie_data = pd.read_csv('movie_dataset_emotions.csv')

# Explore the first few rows of the dataframe
movie_data.head()


# In[43]:


missing_emotion = df['emotion'].isnull().sum()
print(missing_emotion)


# In[44]:


# Checking for unique emotions
unique_emotions = movie_data['emotion'].unique()

# Checking for unique genres, directors, and actors
unique_genres = movie_data['genres'].apply(lambda x: x.split(', ') if pd.notnull(x) else x).explode().unique()
unique_directors = movie_data['directors'].apply(lambda x: x.split(', ') if pd.notnull(x) else x).explode().unique()
unique_actors = movie_data['actors'].apply(lambda x: x.split(', ') if pd.notnull(x) else x).explode().unique()

# Checking for missing values
missing_values = movie_data[['emotion', 'genres', 'directors', 'actors','tomatometer_rating']].isnull().sum()

(unique_emotions, unique_genres, unique_directors, unique_actors, missing_values)


# In[45]:


movie_data_cleaned= movie_data.drop(columns=['directors','actors'])


# In[46]:


# Ensure there are no missing values in 'emotion' and 'genres'
assert movie_data_cleaned[['emotion', 'genres']].isnull().sum().sum() == 0, "There are missing values in the columns to be vectorized."


# # Feature Extraction:

# In[47]:


emotion_ohe = pd.get_dummies(movie_data_cleaned['emotion'], prefix='Emotion')


# In[48]:


user_history = pd.read_csv('cleaned_user_watch_history.csv')

# Explore the first few rows of the dataframe
user_history.head()


# In[49]:


import random

def select_random_by_emotion_and_user(emotion, user_id, dataframe):
    """
    Selects a random row from the dataframe matching the specified emotion and User_ID.
    Returns a vector of the selected row excluding 'Movie Title' and 'User_ID'.

    :param emotion: The emotion to filter by.
    :param user_id: The User_ID to filter by.
    :param dataframe: The pandas DataFrame to search in.
    :return: A vector (list) of the selected row, excluding 'Movie Title' and 'User_ID'.
    """
    # Filtering the dataframe based on emotion and User_ID
    filtered_df = dataframe[(dataframe['Emotion'] == emotion) & (dataframe['User_ID'] == user_id)]

    # If no matching rows found, return None
    if filtered_df.empty:
        return None

    # Selecting a random row
    random_row = filtered_df.sample(n=1).iloc[0]

    # Converting the row to a vector, excluding 'Movie Title' and 'User_ID'
    row_vector = random_row.drop(labels=['Movie Title', 'User_ID']).tolist()

    return row_vector

# Loading the dataset from the uploaded file
uploaded_file_path = 'cleaned_user_watch_history.csv'
uploaded_df = pd.read_csv(uploaded_file_path)

# Example usage of the function
example_emotion = 'Excitement'
example_user_id = 1
random_row_vector = select_random_by_emotion_and_user(example_emotion, example_user_id, uploaded_df)

random_row_vector


# In[50]:




import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def select_random_by_emotion_and_user(new_emotion, user_id, dataframe):
    """
    Selects a random row from the dataframe matching the specified emotion and User_ID.
    Returns a vector of the selected row excluding 'movie_title' and 'User_ID'.
    """
    filtered_df = dataframe[(dataframe['Emotion'] == new_emotion) & (dataframe['User_ID'] == user_id)]

    if filtered_df.empty:
        return None

    random_row = filtered_df.sample(n=1).iloc[0]
    row_vector = random_row.drop(labels=['Movie Title', 'User_ID']).tolist()

    return row_vector

# Testing the functions with the updated datasets

uploaded_file_path = 'cleaned_user_watch_history.csv'
uploaded_df = pd.read_csv(uploaded_file_path)


random_row_vector = select_random_by_emotion_and_user('Excitement', 1, uploaded_df)

def recommend_movies_with_numerical_similarity(dataset, input_vector):
    """
    Recommends the top 10 movies based on combined cosine similarity (text fields) 
    and numerical similarity (rating and count) with the input vector.
    """
    text_vector, numerical_vector = input_vector[:-2], input_vector[-2:]
    dataset['combined_text'] = dataset[['genres', 'directors', 'actors', 'emotion']].apply(lambda x: ' '.join(x.astype(str)), axis=1)

    input_text_str = ' '.join(text_vector)
    appended_dataset = dataset.append({'combined_text': input_text_str}, ignore_index=True)

    vectorizer = CountVectorizer(max_features=10000)
    text_vectors = vectorizer.fit_transform(appended_dataset['combined_text'])

    text_csim = cosine_similarity(text_vectors[-1], text_vectors[:-1])  # Exclude the input vector itself

    # Normalize rating and count
    dataset['norm_rating'] = (dataset['tomatometer_rating'] - dataset['tomatometer_rating'].min()) / (dataset['tomatometer_rating'].max() - dataset['tomatometer_rating'].min())
    dataset['norm_count'] = (dataset['tomatometer_count'] - dataset['tomatometer_count'].min()) / (dataset['tomatometer_count'].max() - dataset['tomatometer_count'].min())

    # Calculate numerical similarity
    numerical_vector_normalized = [
        (numerical_vector[0] - dataset['tomatometer_rating'].min()) / (dataset['tomatometer_rating'].max() - dataset['tomatometer_rating'].min()),
        (numerical_vector[1] - dataset['tomatometer_count'].min()) / (dataset['tomatometer_count'].max() - dataset['tomatometer_count'].min())
    ]
    numerical_distance = np.sqrt((dataset['norm_rating'] - numerical_vector_normalized[0])**2 + (dataset['norm_count'] - numerical_vector_normalized[1])**2)
    numerical_similarity = 1 / (1 + numerical_distance)

    # Combined similarity score
    combined_similarity = (text_csim.flatten() + numerical_similarity) / 2

    top_10_indices = np.argsort(combined_similarity)[::-1][:10]
    top_10_movies = dataset.iloc[top_10_indices]['movie_title'].tolist()

    return top_10_movies

file_path = 'cleaned_filtered_movie_data.csv'
# Load the dataset into a DataFrame
movie_dataset = pd.read_csv(file_path)

# Re-test with the revised function
if random_row_vector is not None:
    top_10_recommended_movies = recommend_movies_with_numerical_similarity(movie_dataset, random_row_vector)
else:
    top_10_recommended_movies = "No matching records found for the given emotion and user ID."

top_10_recommended_movies





# In[65]:


emotions = {"happy":"Neutral",
            "sad":"Excitement",
            "neutral":"Happiness",
            "disgust":"Love",
            "fear":"Excitement",
            "angry" : "Happiness"
    }


# In[67]:


import pyaudio
import wave



#Test the function
def detect_emotion(audio_file):
    # Load the audio file and extract MFCCs
    y, sr = librosa.load(audio_file, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
    
    # Resize and scale the MFCCs
    resized_mfccs = resize_array(mfccs)
    scaled_mfccs = (resized_mfccs - tr_mean) / tr_std
    scaled_mfccs = np.resize(mfccs, (1, 30, 150, 1))

    
    # Predict the emotion using the trained model
    prediction = model_saved.predict(scaled_mfccs)
    emotion_index = np.argmax(prediction)
    
    # Map the emotion index back to the original labels
    emotion_mapping = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad'}
    predicted_emotion = emotion_mapping[emotion_index]
    
    return predicted_emotion


# # # Record audio from user

# CHUNK = 1024 
# FORMAT = pyaudio.paInt16 #paInt8
# CHANNELS = 2 
# RATE = 44100 #sample rate
# RECORD_SECONDS = 5
# WAVE_OUTPUT_FILENAME = "newUsers\\datanew.wav"

# p = pyaudio.PyAudio()

# stream = p.open(format=FORMAT,
#                 channels=CHANNELS,
#                 rate=RATE,
#                 input=True,
#                 frames_per_buffer=CHUNK) #buffer

# print("* recording")

# frames = []

# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#     data = stream.read(CHUNK)
#     frames.append(data) # 2 bytes(16 bits) per channel

# print("* done recording")

# stream.stop_stream()
# stream.close()
# p.terminate()

# wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
# wf.setnchannels(CHANNELS)
# wf.setsampwidth(p.get_sample_size(FORMAT))
# wf.setframerate(RATE)
# wf.writeframes(b''.join(frames))
# wf.close()

# audio_file = "newUsers\\datanew.wav"

#audio_file = 'C:\\Users\\hp\\Downloads\\newUsers\\input_data.wav'


audio_file = 'tess\\OAF_happy\\OAF_met_happy.wav'

predicted_emotion = detect_emotion(audio_file)
print("Predicted emotion:", predicted_emotion)


random_row_vector = select_random_by_emotion_and_user(emotions[predicted_emotion], 2, uploaded_df)

# Re-test with the revised function
if random_row_vector is not None:
    top_10_recommended_movies = recommend_movies_with_numerical_similarity(movie_dataset, random_row_vector)
else:
    top_10_recommended_movies = "No matching records found for the given emotion and user ID."

top_10_recommended_movies


# In[ ]:





# In[ ]:





# In[68]:


import tkinter as tk
from tkinter import messagebox, Toplevel, Label
import threading
import pyaudio
import wave
import audioop

def is_silent(data_chunk, threshold=500):
    """Check if the provided chunk of audio data is below a certain threshold of loudness."""
    return audioop.rms(data_chunk, 2) < threshold

def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "output1.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    frames = []
    silent = True

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        if silent and not is_silent(data):
            silent = False
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    if silent:
        raise ValueError("No speech detected. Please try again.")

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return WAVE_OUTPUT_FILENAME

def process_audio():
    try:
        audio_file = record_audio()
        emotion = detect_emotion(audio_file)
        recommendations = recommend_movies_with_numerical_similarity(movie_dataset, random_row_vector)
        messagebox.showinfo("Recommendations", f"Emotion: {emotion}\nMovies: {recommendations}")
    except ValueError as e:
        messagebox.showerror("Error", str(e))

def start_recording():
    threading.Thread(target=process_audio).start()



def create_loading_screen():
    """
    Creates a small loading screen indicating that processing is happening.
    """
    top = Toplevel(root)
    top.title("Processing")
    top.geometry("300x100")
    Label(top, text="Processing your audio, please wait...", font=("Helvetica", 12)).pack(pady=20)
    # Automatically destroy the top window after 5 seconds
    top.after(5000, top.destroy)

# Rest of your existing code...

# Tkinter GUI Setup
root = tk.Tk()
root.title("Movie Recommendation System")
root.geometry("500x300") # Set the window size

root.configure(bg='#f0f0f0') # Light grey background

# Attractive Button Design
record_button = tk.Button(root, text="Record and Get Recommendations", command=start_recording,
                          bg="#007bff", fg="white", padx=20, pady=10, font=("Helvetica", 16))
record_button.pack(pady=50)

# Add a footer label for additional info or credits
footer_label = tk.Label(root, text="Powered by SRS", bg='#f0f0f0', font=("Helvetica", 10))
footer_label.pack(side="bottom", fill="x")

root.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Movies Recommendations System

# In[ ]:


pip install joblib


# In[56]:


import pandas as pd

movies_data = pd.read_csv('Movies_dataset.csv')

# Display the first few rows of the dataset
movies_data.head()


# # Preprocessing and labeling Data

# # Labeling on the base of Weightage

# In[57]:


# Define weights for each genre's emotion
emotion_weights = {
    "Drama": 0.6,
    "Tragedy": 0.6,
    "Classics": 0.6,
    "Comedy": 0.5,
    "Animation": 0.5,
    "Musical & Performing Arts": 0.5,
    "Horror": 0.7,
    "Mystery & Suspense": 0.7,
    "Romance": 0.5,
    "Documentary": 0.3,
    "Art House & International": 0.3,
    "Special Interest": 0.3,
    "Action & Adventure": 0.4,
    "Science Fiction & Fantasy": 0.4
}


def label_emotion(genre):
    if genre in ["Drama", "Tragedy", "Classics"]:
        return "Sadness"
    elif genre in ["Comedy", "Animation", "Musical & Performing Arts"]:
        return "Happiness"
    elif genre in ["Horror", "Mystery & Suspense"]:
        return "Fear"
    elif genre == "Romance":
        return "Love"
    elif genre in ["Documentary", "Art House & International", "Special Interest"]:
        return "Neutral"
    elif genre in ["Action & Adventure", "Science Fiction & Fantasy"]:
        return "Excitement"
    else:
        return "Other"



def calculate_emotion(genres):
    # Split genres and calculate the total weight for each emotion
    emotion_scores = {}
    for genre in genres.split(", "):
        if genre in emotion_weights:
            emotion = label_emotion(genre)  # Get the emotion for the genre using the previous function
            emotion_scores[emotion] = emotion_scores.get(emotion, 0) + emotion_weights[genre]
    
    # Return the emotion with the highest score
    return max(emotion_scores, key=emotion_scores.get, default="Other")

# Apply the calculate_emotion function to the 'genres' column to create the 'emotion' column
movies_data['emotion'] = movies_data['genres'].apply(calculate_emotion)

# Display the first few rows to see the updated 'emotion' column
movies_data[['movie_title', 'genres', 'emotion']].head()


# In[58]:


output_path = "movie_dataset_emotions.csv"
movies_data.to_csv(output_path, index=False)


# In[59]:


data = pd.read_csv('movie_dataset_emotions.csv')
data


# # Vectorization

# In[60]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[61]:


# 1. Drop rows where 'genres', 'directors', or 'emotions' have missing values
data = data.dropna(subset=['genres', 'directors', 'emotion', 'actors'])

# 2. Fill missing values in the 'actors' column with 'Unknown'
data['actors'].fillna('Unknown', inplace=True)

# 3. Combine 'genres', 'directors', 'actors', and 'emotions' columns for vectorization
data['combined_features'] = data['genres'] + ' ' + data['directors'] + ' ' + data['actors'] + ' ' + data['emotion']

data['combined_features'].head()


# In[62]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
print(vectorizer)
# Fit and transform combined_features to get the matrix
count_matrix = vectorizer.fit_transform(data['combined_features'])
print(count_matrix)
# Compute the cosine similarity matrix from the count_matrix
cosine_sim = cosine_similarity(count_matrix)

cosine_sim.shape


# In[ ]:





# In[63]:


data["emotion"].unique()




# data[data['emotion'] == 'Happiness']


# In[123]:



emotions = {"happy":"Neutral",
            "sad":"Excitement",
            "neutral":"Happiness",
            "disgust":"Love",
            "fear":"Excitement"        
    }



# In[91]:


# from sklearn.metrics.pairwise import cosine_similarity


# import random

# def select_random_by_emotion_and_user(emotion, user_id, dataframe):
#     """
#     Selects a random row from the dataframe matching the specified emotion and User_ID.
#     Returns a vector of the selected row excluding 'Movie Title' and 'User_ID'.

#     :param emotion: The emotion to filter by.
#     :param user_id: The User_ID to filter by.
#     :param dataframe: The pandas DataFrame to search in.
#     :return: A vector (list) of the selected row, excluding 'Movie Title' and 'User_ID'.
#     """
#     # Filtering the dataframe based on emotion and User_ID
#     filtered_df = dataframe[(dataframe['Emotion'] == emotion) & (dataframe['User_ID'] == user_id)]

#     # If no matching rows found, return None
#     if filtered_df.empty:
#         return None

#     # Selecting a random row
#     random_row = filtered_df.sample(n=1).iloc[0]

#     # Converting the row to a vector, excluding 'Movie Title' and 'User_ID'
#     row_vector = random_row.drop(labels=['Movie Title', 'User_ID']).tolist()

#     return row_vector


# In[53]:


uploaded_file_path = 'cleaned_user_watch_history.csv'
uploaded_df = pd.read_csv(uploaded_file_path)



random_row_vector=select_random_by_emotion_and_user('Fear', 1, uploaded_df)

print(random_row_vector)


# In[ ]:





# In[35]:




import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def select_random_by_emotion_and_user(new_emotion, user_id, dataframe):
    """
    Selects a random row from the dataframe matching the specified emotion and User_ID.
    Returns a vector of the selected row excluding 'movie_title' and 'User_ID'.
    """
    filtered_df = dataframe[(dataframe['Emotion'] == new_emotion) & (dataframe['User_ID'] == user_id)]

    if filtered_df.empty:
        return None

    random_row = filtered_df.sample(n=1).iloc[0]
    row_vector = random_row.drop(labels=['Movie Title', 'User_ID']).tolist()

    return row_vector

# Testing the functions with the updated datasets

uploaded_file_path = 'cleaned_user_watch_history.csv'
uploaded_df = pd.read_csv(uploaded_file_path)


random_row_vector = select_random_by_emotion_and_user('Excitement', 1, uploaded_df)

def recommend_movies_with_numerical_similarity(dataset, input_vector):
    """
    Recommends the top 10 movies based on combined cosine similarity (text fields) 
    and numerical similarity (rating and count) with the input vector.
    """
    text_vector, numerical_vector = input_vector[:-2], input_vector[-2:]
    dataset['combined_text'] = dataset[['genres', 'directors', 'actors', 'emotion']].apply(lambda x: ' '.join(x.astype(str)), axis=1)

    input_text_str = ' '.join(text_vector)
    appended_dataset = dataset.append({'combined_text': input_text_str}, ignore_index=True)

    vectorizer = CountVectorizer(max_features=10000)
    text_vectors = vectorizer.fit_transform(appended_dataset['combined_text'])

    text_csim = cosine_similarity(text_vectors[-1], text_vectors[:-1])  # Exclude the input vector itself

    # Normalize rating and count
    dataset['norm_rating'] = (dataset['tomatometer_rating'] - dataset['tomatometer_rating'].min()) / (dataset['tomatometer_rating'].max() - dataset['tomatometer_rating'].min())
    dataset['norm_count'] = (dataset['tomatometer_count'] - dataset['tomatometer_count'].min()) / (dataset['tomatometer_count'].max() - dataset['tomatometer_count'].min())

    # Calculate numerical similarity
    numerical_vector_normalized = [
        (numerical_vector[0] - dataset['tomatometer_rating'].min()) / (dataset['tomatometer_rating'].max() - dataset['tomatometer_rating'].min()),
        (numerical_vector[1] - dataset['tomatometer_count'].min()) / (dataset['tomatometer_count'].max() - dataset['tomatometer_count'].min())
    ]
    numerical_distance = np.sqrt((dataset['norm_rating'] - numerical_vector_normalized[0])**2 + (dataset['norm_count'] - numerical_vector_normalized[1])**2)
    numerical_similarity = 1 / (1 + numerical_distance)

    # Combined similarity score
    combined_similarity = (text_csim.flatten() + numerical_similarity) / 2

    top_10_indices = np.argsort(combined_similarity)[::-1][:10]
    top_10_movies = dataset.iloc[top_10_indices]['movie_title'].tolist()

    return top_10_movies

file_path = 'cleaned_filtered_movie_data.csv'
# Load the dataset into a DataFrame
movie_dataset = pd.read_csv(file_path)

# Re-test with the revised function
if random_row_vector is not None:
    top_10_recommended_movies = recommend_movies_with_numerical_similarity(movie_dataset, random_row_vector)
else:
    top_10_recommended_movies = "No matching records found for the given emotion and user ID."

top_10_recommended_movies





# In[36]:


movie_dataset.head()


# In[39]:


import pyaudio
import wave



#Test the function
def detect_emotion(audio_file):
    # Load the audio file and extract MFCCs
    y, sr = librosa.load(audio_file, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
    
    # Resize and scale the MFCCs
    resized_mfccs = resize_array(mfccs)
    scaled_mfccs = (resized_mfccs - tr_mean) / tr_std
    scaled_mfccs = np.resize(mfccs, (1, 30, 150, 1))

    
    # Predict the emotion using the trained model
    prediction = model_saved.predict(scaled_mfccs)
    emotion_index = np.argmax(prediction)
    
    # Map the emotion index back to the original labels
    emotion_mapping = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad'}
    predicted_emotion = emotion_mapping[emotion_index]
    
    return predicted_emotion


# # Record audio from user

CHUNK = 1024 
FORMAT = pyaudio.paInt16 #paInt8
CHANNELS = 2 
RATE = 44100 #sample rate
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "newUsers\\datanew.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK) #buffer

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data) # 2 bytes(16 bits) per channel

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

audio_file = "newUsers\\datanew.wav"

#audio_file = 'C:\\Users\\hp\\Downloads\\newUsers\\input_data.wav'


#audio_file = 'tess\\OAF_disgust\\OAF_met_disgust.wav'

predicted_emotion = detect_emotion(audio_file)
print("Predicted emotion:", predicted_emotion)


random_row_vector = select_random_by_emotion_and_user(emotions[predicted_emotion], 1, uploaded_df)

# Re-test with the revised function
if random_row_vector is not None:
    top_10_recommended_movies = recommend_movies_with_numerical_similarity(movie_dataset, random_row_vector)
else:
    top_10_recommended_movies = "No matching records found for the given emotion and user ID."

top_10_recommended_movies


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Defining the Classifiers

# In[82]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder


# In[83]:


# Load your dataset
data = pd.read_csv("movie_dataset_emotions.csv")
#print(data['genres'])
# Preprocessing
data['genres'] = data['genres'].str.split(', ')
#print(data['genres'])
# Use MultiLabelBinarizer for genres
mlb = MultiLabelBinarizer()
genres_encoded = pd.DataFrame(mlb.fit_transform(data['genres']), columns=mlb.classes_)
#print(genres_encoded)


# In[84]:


# Encoding emotions using LabelEncoder
label_encoder = LabelEncoder()
data["emotion"] = label_encoder.fit_transform(data["emotion"])
print(data["emotion"])

data = pd.concat([data['emotion'], genres_encoded], axis=1)  # Keep only 'emotions' and genres columns
#print(genres_encoded)

#print(data)
# Split dataset
X = data.drop(['emotion'], axis=1)  # Input features: only genres

y = data['emotion']  # Target variable: emotions
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Random Forest

# In[85]:


# # RANDOM FOREST

# In[18]:


# Train a Random Forest classifier
from sklearn.metrics import classification_report

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Step 8: Print the classification report
print(classification_rep)


# # SVM

# In[86]:


from sklearn.svm import SVC
#Build the SVM classifier
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

#Evaluate the SVM model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Step 8: Print the classification report
print(classification_rep)


# # Gradient Boosting Classifier

# In[87]:


from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Step 6: Evaluate the Gradient Boosting model
y_pred = gb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Step 8: Print the classification report
print(classification_rep)


# # Naive Bayes Classifier

# In[88]:


from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Step 8: Print the classification report
print(classification_rep)


# # Decision Tree Classifier

# In[89]:


from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Step 8: Print the classification report
print(classification_rep)


# # Logistic Regression

# In[90]:


from sklearn.linear_model import LogisticRegression
logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_model.fit(X_train, y_train)

# Step 6: Evaluate the Logistic Regression model
y_pred = logreg_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Step 8: Print the classification report
print(classification_rep)


# In[91]:


from sklearn.model_selection import cross_val_score

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import MultiLabelBinarizer


# In[92]:


# List of classifiers to cross-validate
classifiers = [
    ("Random Forest", RandomForestClassifier(random_state=42)),
    ("SVM", SVC(random_state=42)),
    ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
    ("Naive Bayes", GaussianNB()),
    ("Gradient Boosting", GradientBoostingClassifier()),
    ("Decision Tree Classifier", DecisionTreeClassifier()),
]

# Pserform cross-validation for each classifier
for clf_name, clf in classifiers:
    scores = cross_val_score(clf, X, y, cv=5)  # 5-fold cross-validation
    print(f"{clf_name} - Accuracy: {scores.mean():.4f} (std: {scores.std():.4f})")


# In[ ]:





# In[ ]:





# In[ ]:




