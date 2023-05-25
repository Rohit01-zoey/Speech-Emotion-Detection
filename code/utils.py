import pandas as pd
import numpy as np
import os
import opensmile
import librosa
import joblib

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
    def __init__(self, data, speaker_ids, utt_ids):
        self.data = data
        self.speaker_ids = speaker_ids
        self.utt_ids = utt_ids

    def get_data_by_speaker(self, speaker_id):
        mask = [speaker == speaker_id for speaker in self.speaker_ids]
        return [item for item, m in zip(self.data, mask) if m]

    def get_data_by_utt(self, utt_id):
        mask = [utt == utt_id for utt in self.utt_ids]
        return [item for item, m in zip(self.data, mask) if m]

    def get_data_by_speaker_utt(self, speaker_id, utt_id):
        mask = [(speaker == speaker_id) and (utt == utt_id) for speaker, utt in zip(self.speaker_ids, self.utt_ids)]
        return [item for item, m in zip(self.data, mask) if m]


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