# 1155202866
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# Problem 2
def problem_2(df, k=5, t=0.2):
    list = []
    train, test = train_test_split(df, test_size=t)
    list_size = len(train) // k
    reminder = len(train) % k
    for i in range(k):
        if i < reminder:
            list.append(train[i * (list_size + 1):(i + 1) * (list_size + 1)])
        else:
            list.append(train[i * list_size + reminder:(i + 1) * list_size + reminder])
    return list, test

# Problem 3
import torchaudio
from torchaudio import transforms
import torch
def problem_3(audio_object, new_sr, max_mask_pct, n_freq_masks,n_time_masks):
    sig,sr = audio_object
    top_db = 80
    n_mels = 64
    n_fft = 1024
    hop_len = None
    aug_spec = 0 # it should be the augmented mel spectrogram
    # write your logic here

    # Standardize Sampling Rate
    if (sr != new_sr):
        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, new_sr)(sig[:1,:])
        if (num_channels > 1 and 1 == 2):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, new_sr)(sig[1:,:])
            resig = torch.cat([resig, retwo])
        sig = resig
        sr = new_sr
    
    # Mel Spectrogram
    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    
    # Time and Frequency Masking
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec
    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
        aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
        aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
    return aug_spec

# Problem 4
def problem_4(input_mat):
    output_mat = np.array([0])
    # write your logic here
    for i in range(input_mat.shape[0]):
        for j in range(input_mat.shape[1]):
            output_mat = np.append(output_mat, input_mat[i][j])
        
    return output_mat

# Problem 5
import re
def problem_5(df_input):
    df_output = df_input
    # write your logic here
    for i in range(df_output.shape[0]):
        error = False
        number = df_output.iloc[i,0]
        if '1' <= number[0] <= '9':
            if len(number) == 9:
                if number[4] == '-':
                    number = number.replace('-', '')
            if len(number) == 8:
                for k in range(8):
                    if number[k] < '0' or number[k] > '9':
                        error = True
            else:
                error = True
        else:
            error = True
        if error == False:
            df_output.iloc[i,0] = number
        else:
            df_output.iloc[i,0] = 'ERROR'
    return df_output

# Problem 6
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
def problem_6(df_input, label, k1=10, k2=10):
    df_output = df_input
    # write your logic here

    X = df_output.drop(label, axis=1)
    y = df_output[label]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=k1)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(data = X_pca, columns = [i for i in range(k1)])

    lda = LDA(n_components=k2)
    X_lda = lda.fit_transform(X_scaled, y)
    df_lda = pd.DataFrame(data = X_lda, columns = [i+k1 for i in range(k2)])

    df_output = pd.concat([df_pca, df_lda], axis=1)
    
    
    return df_output

if __name__ == "__main__":
    # Testing: Problem 2
    df = pd.read_csv("mlb_players.csv")
    list, test = problem_2(df, k=5, t=0.2)
    for item in list:
        print("Segment: ", item.shape)
    print("Testing: ", test.shape)

    # Testing: Problem 3
    audio_file = "greensleeves.mp3"
    sig, sr = torchaudio.load(audio_file)
    # first 10 second
    sig = sig[:,:10*sr]
    new_sr, max_mask_pct, n_freq_masks, n_time_masks = 1024, 0.1, 1, 1
    aug_spec = problem_3((sig,sr), new_sr, max_mask_pct, n_freq_masks, n_time_masks)
    print(aug_spec)
    print(aug_spec.shape)
    
    # Testing: Problem 4
    input_mat = np.array([[1,2,3],[4,5,6]])
    output_mat = problem_4(input_mat)
    print("Input", input_mat)
    print("Output", output_mat)
    
    # Testing: Problem 5
    list_of_number = ["1234-5678", "0123-4567", "1234567", "8888-4444", "98983434"]
    dict = {"phone":list_of_number}
    df = pd.DataFrame(dict)
    df_result = problem_5(df)
    print(df_result)
    print(df_result.shape)
    
    # Testing: Problem 6
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
    dataset = pd.read_csv(url, names=names)
    df_feature = problem_6(dataset, 'Class', 3, 1)
    print(df_feature)
    print(df_feature.shape)
