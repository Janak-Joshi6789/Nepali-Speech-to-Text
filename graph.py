
### **Steps to Generate the Graphs**


### **1. Prepare the Dataset and Model**

###Make sure your **test dataset** and **trained model** are ready.

###1. **Load the model**:

   from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
   import torch

   # Load pretrained model and processor
   model = Wav2Vec2ForCTC.from_pretrained("path_to_your_model")
   processor = Wav2Vec2Processor.from_pretrained("path_to_your_processor")

#2. **Prepare your test dataset**:
#   - Make sure your test dataset contains the **audio files** and their **true transcriptions**.
#   - You can load and preprocess the audio data using libraries like **`librosa`** or **`torchaudio`**.
 #  - For example, loading an audio file might look like this:

 #  ```python
   import librosa

   # Load an audio file
   audio_path = "path_to_audio_file.wav"
   audio_input, _ = librosa.load(audio_path, sr=16000)  # Assuming 16kHz sample rate



### **2. Predict Transcriptions Using Your Model**

#1. **Audio Transcription**:
 #  - Use the model to transcribe the audio.
   
 #  ```python
   # Preprocess the audio input
   input_values = processor(audio_input, return_tensors="pt").input_values

   # Make prediction
   with torch.no_grad():
       logits = model(input_values).logits

   # Convert logits to predicted tokens (ids)
   predicted_ids = torch.argmax(logits, dim=-1)
   transcription = processor.decode(predicted_ids[0])
   


### **3. Calculate WER and CER**


#1. **Install the `jiwer` library** (for WER) if you haven't already:

   pip install jiwer

#2. **Calculate WER and CER** for each test sample (compare the model’s transcription with the ground truth transcription).

   from jiwer import wer, cer

   # Example ground truth and predicted transcription
   ground_truth = "यो एउटा परीक्षण वाक्य हो।"
   predicted = transcription

   # Compute WER and CER
   wer_score = wer(ground_truth, predicted)
   cer_score = cer(ground_truth, predicted)

   print(f"WER: {wer_score}, CER: {cer_score}")


#3. **Store WER and CER** for all test samples in a list or DataFrame for plotting.


### **4. Create Graphs**


### **4.1 Audio Duration Distribution (Histogram)**

#1. **Get the durations** of the test audio files.
#2. **Plot** the histogram.

import matplotlib.pyplot as plt

# Example: Assuming you have a list of audio durations
audio_durations = [2.5, 3.0, 5.1, 4.3, 1.8]  # Replace with actual durations

# Plotting histogram
plt.hist(audio_durations, bins=15, color="blue", alpha=0.7)
plt.xlabel("Audio Duration (seconds)")
plt.ylabel("Frequency")
plt.title("Distribution of Audio Durations")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


### **4.2 Word Frequency Distribution (Bar Chart)**

1. **Tokenize the transcriptions** to count word frequencies.
2. **Plot the bar chart** for the top 10 most frequent words.

from collections import Counter
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Combine all transcriptions
all_transcriptions = ["यो एउटा परीक्षण वाक्य हो।", "हामी नेपाली हौं।"]  # Replace with your actual transcriptions

# Tokenize words and count frequencies
words = word_tokenize(" ".join(all_transcriptions))
word_counts = Counter(words)
common_words = word_counts.most_common(10)

# Plot bar chart
words, counts = zip(*common_words)
plt.bar(words, counts, color="green")
plt.xticks(rotation=45)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 10 Most Frequent Words in Transcriptions")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


### **4.3 WER & CER Comparison (Bar Chart)**

1. **Plot WER and CER values** for each test sample or model variant.

# Assuming you have WER and CER values for each model
wer_model_values = [0.21, 0.18, 0.15]  # Replace with actual values
cer_model_values = [0.10, 0.08, 0.06]  # Replace with actual values
models = ["Baseline", "Fine-tuned", "Optimized"]

x = range(len(models))
plt.bar(x, wer_model_values, width=0.4, label="WER", color="blue", align="center")
plt.bar(x, cer_model_values, width=0.4, label="CER", color="orange", align="edge")

plt.xticks(x, models)
plt.ylabel("Error Rate")
plt.title("WER and CER Comparison Across Models")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


### **4.4 WER vs Audio Duration (Scatter Plot)**

#1. **Plot a scatter plot** with audio duration on the x-axis and WER on the y-axis.

# Example data: durations and WER scores for each audio sample
audio_durations = [2.5, 3.0, 5.1, 4.3, 1.8]  # Replace with actual durations
wer_scores = [0.15, 0.20, 0.18, 0.25, 0.30]  # Replace with actual WER values

plt.scatter(audio_durations, wer_scores, color="purple", alpha=0.5)
plt.xlabel("Audio Duration (seconds)")
plt.ylabel("Word Error Rate (WER)")
plt.title("WER vs Audio Duration")
plt.grid(True)
plt.show()

