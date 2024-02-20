#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re

def detect_filled_pauses(text):
    filled_pauses = ['uh', 'um']
    for pause in filled_pauses:
        # Use regular expression to find occurrences of the filled pause in the text
        matches = re.findall(r'\b{}\b'.format(pause), text)
        if matches:
            return "bad"
    return "good"

# Example text
speech_text = "input here text data of audio umm uhh uh"

# Detect filled pauses in the speech text
fluency = detect_filled_pauses(speech_text.lower())

# Print the overall fluency classification
print("Fluency classification:", fluency)


# In[30]:


import librosa
import numpy as np

audio_file = "C:/Users/windows 11/Documents/video_audio3.mp3"

y, sr = librosa.load(audio_file, sr=None)

pitch_values, _ = librosa.piptrack(y=y, sr=sr)

pitch_values_mean = pitch_values.mean(axis=0)

threshold_low_variation = 20  # Hz
threshold_high_variation = 40  # Hz
min_duration_low_variation = 3  # seconds
min_duration_high_variation = 0.2  # seconds

low_pitch_count = 0
high_pitch_count = 0

in_low_variation = False
in_high_variation = False

for pitch_value in pitch_values_mean:
    if pitch_value < threshold_low_variation:
        if not in_low_variation:
            in_low_variation = True
            low_pitch_count += 1
            in_high_variation = False
    elif pitch_value > threshold_high_variation:
        if not in_high_variation:
            in_high_variation = True
            high_pitch_count += 1
            in_low_variation = False
    else:
        in_low_variation = False
        in_high_variation = False
fluency= ((high_pitch_count/low_pitch_count)*100)

print("fluecny Score ", fluency)


# In[32]:


import librosa
import matplotlib.pyplot as plt

# Load audio file (replace 'path/to/audio/file.wav' with your file path)
audio_file = "C:/Users/windows 11/Downloads/titanium-170190.mp3"

# Load the audio file and the sampling rate
y, sr = librosa.load(audio_file, sr=None)

# Estimate pitch using librosa
pitch_values, _ = librosa.piptrack(y=y, sr=sr)

# Extract pitch values from the first channel and take the mean across time
pitch_values_mean = pitch_values.mean(axis=0)

# Create a time array for plotting
time = librosa.times_like(pitch_values_mean, sr=sr)

# Plot the pitch values
plt.figure(figsize=(10, 4))
plt.plot(time, pitch_values_mean, label='Pitch (Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Pitch Estimation')
plt.ylim(50, 200)  # Set y-axis limits to the range of 50 Hz to 200 Hz
plt.legend()
plt.grid()
plt.show()


# In[3]:


import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio file (replace 'path/to/audio/file.wav' with your file path)
audio_file = "C:/Users/windows 11/Downloads/titanium-170190.mp3"

# Load the audio file and the sampling rate
y, sr = librosa.load(audio_file)

# Compute the spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr)

# Convert to decibel (log scale)
S_db = librosa.power_to_db(S, ref=np.max)

# Plot the spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='linear')  # Use 'linear' for frequency axis
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()


# In[6]:


# Define hop length (typically used in librosa for computing spectrograms)
hop_length = 512  # You can adjust this value based on your preference

# Load audio file and compute spectrogram
y, sr = librosa.load(audio_file)
S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)

# Calculate fluency score for the entire spectrogram
fluency_score = calculate_fluency_score(S)

# Thresholding: classify segments as fluent or disfluent
fluent_segments = np.where(fluency_score > FLUENT_THRESHOLD)[0]
disfluent_segments = np.where(fluency_score <= FLUENT_THRESHOLD)[0]

# Plot the spectrogram
plt.figure(figsize=(10, 6))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram with Fluency Score')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
import numpy as np

def calculate_fluency_score(spectrogram):
    """
    Calculate fluency score based on the spectrogram
    
    Args:
    - spectrogram: 2D numpy array, spectrogram of the audio signal
    
    Returns:
    - fluency_score: 1D numpy array, fluency score for each time frame
    """
    # Example of a simple fluency scoring criterion:
    # Calculate the mean amplitude of each time frame
    frame_amplitudes = np.mean(spectrogram, axis=0)
    
    # Scale the amplitude values to the range [0, 1]
    scaled_amplitudes = (frame_amplitudes - np.min(frame_amplitudes)) / (np.max(frame_amplitudes) - np.min(frame_amplitudes))
    
    # Invert the amplitude values so that higher amplitudes indicate fluent segments
    fluency_score = 1 - scaled_amplitudes
    
    return fluency_score

# Example usage:
fluency_score = calculate_fluency_score(S)
print("Fluency Score:", fluency_score)


# Annotate fluent segments
for segment in fluent_segments:
    plt.axvspan(segment * hop_length / sr, (segment + 1) * hop_length / sr, color='green', alpha=0.3)

# Annotate disfluent segments
for segment in disfluent_segments:
    plt.axvspan(segment * hop_length / sr, (segment + 1) * hop_length / sr, color='red', alpha=0.3)

plt.show()


# In[8]:


def calculate_single_fluency_score(spectrogram):
    
    # Calculate the fluency score for each time frame
    fluency_scores = calculate_fluency_score(spectrogram)
    
    # Aggregate the fluency scores to get a single fluency score
    single_fluency_score = np.mean(fluency_scores)
    
    return single_fluency_score

# Example usage:
single_fluency_score = calculate_single_fluency_score(S)
print("Fluency Score:", single_fluency_score)

