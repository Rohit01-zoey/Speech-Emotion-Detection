import pandas as pd
import numpy as np
import os
import opensmile
import librosa
import joblib
from sklearn.model_selection import LeaveOneGroupOut

class StandardScaleNormalizer:
    def __init__(self):
        self.mean = {}
        self.scale = {}

    def fit(self, X, speaker):
        """
        Fits the normalization on input data with respect to each speaker.

        Args:
            X (numpy.ndarray or pandas.DataFrame): Input data.
            speaker (numpy.ndarray or list): Speaker labels.
        """
        df = pd.DataFrame(X)
        df["speakerID"] = speaker

        for id in df["speakerID"].unique():
            speaker_data = df.loc[df["speakerID"] == id].drop("speakerID", axis=1)
            self.mean[id] = speaker_data.mean()
            self.scale[id] = speaker_data.std()

    def transform(self, X, speaker):
        """
        Applies the normalization on input data.

        Args:
            X (numpy.ndarray or pandas.DataFrame): Input data.
            speaker (numpy.ndarray or list): Speaker labels.

        Returns:
            pandas.DataFrame: Normalized data.
        """
        df = pd.DataFrame(X)
        df["speakerID"] = speaker

        df_copy = df.copy()
        for id in df_copy["speakerID"].unique():
            mask = df_copy["speakerID"] == id
            df_copy.loc[mask] = (df_copy.loc[mask] - self.mean[id]) / self.scale[id]

        return df_copy.drop("speakerID", axis=1)

    def fit_transform(self, X, speaker):
        """
        Fits the normalization and applies it on input data.

        Args:
            X (numpy.ndarray or pandas.DataFrame): Input data.
            speaker (numpy.ndarray or list): Speaker labels.

        Returns:
            pandas.DataFrame: Normalized data.
        """
        self.fit(X, speaker)
        return self.transform(X, speaker)

class CustomDataLoader:
    def __init__(self, data, speaker_ids, emotion_ids, utt_ids):
        self.data = data
        self.speaker_ids = speaker_ids
        self.emotion_ids = emotion_ids
        self.utt_ids = utt_ids

    def get_data_by_speaker(self, speaker_id):
        mask = [speaker == speaker_id for speaker in self.speaker_ids]
        return [item for item, m in zip(self.data.to_numpy(), mask) if m]

    def get_data_by_utt(self, utt_id):
        mask = [utt == utt_id for utt in self.utt_ids]
        return [item for item, m in zip(self.data.to_numpy(), mask) if m]

    def get_data_by_speaker_utt(self, speaker_id, utt_id):
        mask = [(speaker == speaker_id) and (utt == utt_id) for speaker, utt in zip(self.speaker_ids, self.utt_ids)]
        return [item for item, m in zip(self.data.to_numpy(), mask) if m]


def load_wav_features(directory):
    """
    Loads the features of WAV files from a directory and its subfolders.

    Args:
        directory (str): Path to the directory containing the WAV files.

    Returns:
        features (np.ndarray): Array of shape (n_samples, n_features) containing the loaded features.
        subfolder_names (list): List of subfolder names corresponding to each WAV file.
        file_names (list): List of file names of the loaded WAV files.

    Raises:
        ValueError: If the directory does not exist or is empty.
        IOError: If there is an error reading the WAV files.

    Notes:
        - The function assumes that the WAV files are located in the directory and its subfolders.
        - The function uses the Opensmile process_signal function to generate features for each WAV file.

    Example:
        directory = 'path/to/wav_files/'
        features, subfolder_names, file_names = load_wav_features(directory)
    """
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
    subfolder_names = []
    file_names = []
    sample_rates = []
    features_array = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                subfolder_name = os.path.basename(root)
                subfolder_names.append(subfolder_name)
                file_names.append(file)
                
                data, sample_rate = librosa.load(file_path)
                
                sample_rates.append(sample_rate)
                
                features = list(smile.process_signal(data, sample_rate).to_numpy())[0]

                # Convert features_list to a 2D array
                features_array.append(features)

    
    
    
    return np.array(features_array), subfolder_names, file_names, sample_rates

def load_svm_ensemble(model_path):
    '''
    All saved models withing the specified path should be of the format 'svc_model_<speaker_id>.pkl
    '''
    trained_model_ensemble = []
    for i in os.listdir(model_path):
        print("Loading model with {spk} left out while training".format(spk=i[10:-4]))
        trained_model_ensemble.append(joblib.load(model_path+i))
    return trained_model_ensemble

def process_ravdess(file_names : list):
    labels_dict = {'01' : 'neutral', '02' : 'calm', '03' : 'happy', '04' : 'sad', '05' : 'angry', '06' : 'fearful', '07' : 'disgust', '08' : 'surprised'}
    utt_dict = {'01':"kids", '02':"dogs"}
    emotions = []
    utt_ids = []
    for file in file_names:
        emotions.append(labels_dict[file.split('-')[2]])
        utt_ids.append(utt_dict[file.split('-')[4]])
    return emotions, utt_ids

def process_hindi(file_names : list):
    # labels_dict = {'01' : 'neutral', '02' : 'calm', '03' : 'happy', '04' : 'sad', '05' : 'angry', '06' : 'fearful', '07' : 'disgust', '08' : 'surprised'}
    # utt_dict = {'01':"kids", '02':"dogs"}
    emotions = []
    utt_ids = []
    for file in file_names:
        emotions.append(file.split('.')[0].split('_')[3])
        utt_ids.append(file.split('.')[0].split('_')[4])
    return emotions, utt_ids

def process_savee(file_names : list):
    emotion = []
    for file in file_names:
        ele = file[:-6]
        if ele=='a':
            emotion.append('angry')
        elif ele=='d':
            emotion.append('disgust')
        elif ele=='f':
            emotion.append('fear')
        elif ele=='h':
            emotion.append('happy')
        elif ele=='n':
            emotion.append('neutral')
        elif ele=='sa':
            emotion.append('sad')
        else:
            emotion.append('surprise')
    return emotion
    

def process_emodb(file_names : list):
    labels_dict = {'W' : 'angry', 'L' : 'boredom', 'E' : 'disgust', 'A' : 'fearful', 'F' : 'happy', 'T' : 'sad', 'N' : 'neutral'}
    utt_dict = {'03':"kids", '08':"dogs"}
    emotions = []
    utt_ids = []
    for file in file_names:
        emotions.append(labels_dict[file.split('.')[0].split('_')[2]])
        utt_ids.append(utt_dict[file.split('.')[0].split('_')[0]])
    return emotions, utt_ids


#write a function which takes in dataloader class as above and returns an iterator which leaves out one speaker at a time. It should take in features, emoitons_list etc use leaveonegropout
def leave_one_speaker_out(dataloader):
    logo = LeaveOneGroupOut()
    for i, (train_index, test_index) in enumerate(logo.split(X=dataloader.data, groups=dataloader.speaker_ids)):
        # print("TRAIN:", train_index, "TEST:", test_index)
        data_train, data_test = np.array(dataloader.data)[train_index], np.array(dataloader.data)[test_index]
        emotion_train, emotion_test = np.array(dataloader.emotion_ids)[train_index], np.array(dataloader.emotion_ids)[test_index]
        utt_train, utt_test = np.array(dataloader.utt_ids)[train_index], np.array(dataloader.utt_ids)[test_index]
        speaker_train, speaker_test = np.array(dataloader.speaker_ids)[train_index], np.array(dataloader.speaker_ids)[test_index]
        yield data_train, data_test, emotion_train, emotion_test, utt_train, utt_test, speaker_train, speaker_test



# write a code for leaving one speaker out and training a model on the rest of the speakers the user should be able to pass a param_grid to select best paraemters
def train_svm_ensemble(features, labels, speaker_ids, utt_ids, model_path, param_grid, n_jobs=1):
    '''
    features : numpy array of features
    labels : list of labels
    speaker_ids : list of speaker ids
    utt_ids : list of utt_ids
    model_path : path to save the trained models
    param_grid : param_grid for the SVM
    '''
    # get unique speaker ids
    unique_speaker_ids = np.unique(speaker_ids)
    # get unique utt_ids
    unique_utt_ids = np.unique(utt_ids)
    # create a custom dataloader
    dataloader = CustomDataLoader(features, speaker_ids, utt_ids)
    # create a leave one speaker out cross validation
    loo = LeaveOneOut()
    # create a grid search object
    grid_search = GridSearchCV(SVC(), param_grid, cv=loo, n_jobs=n_jobs)
    # iterate over unique speaker ids
    for speaker_id in unique_speaker_ids:
        # get data for the current speaker
        X = dataloader.get_data_by_speaker(speaker_id)
        # get labels for the current speaker
        y = labels[speaker_ids == speaker_id]
        # fit the grid search object
        grid_search.fit(X, y)
        # save the model
        joblib.dump(grid_search.best_estimator_, model_path+'svc_model_'+str(speaker_id)+'.pkl')
        print("Saved model with {spk} left out while training".format(spk=speaker_id))
 