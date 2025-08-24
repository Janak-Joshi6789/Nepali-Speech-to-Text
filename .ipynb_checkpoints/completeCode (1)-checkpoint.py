!pip install jiwer
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)
from datasets import Audio, ClassLabel, load_dataset, load_metric
from IPython.display import display, HTML
from sklearn.model_selection import train_test_split
from torch._C import device
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    trainer_utils
)
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import IPython.display as ipd
import jiwer
import librosa
import numpy as np
import pandas as pd
import random
import re
import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import torchaudio
import wandb

wandb.login(key="f8309bfa7d3b1d6d877d79040b6d30b6dc26afcb") # add wandb API key
wandb.init(project="speech")

!mkdir csv_files
!mkdir trained_models
!mkdir checkpoints
!mkdir tokenizer
!mkdir processor 
!mkdir feature_extractor

csv_folder_path = './csv_files'
trained_model_path = './trained_models/wav2vec0.8dropout'
processor_path = './processor/wav2vec0.8dropout'
checkpoints_path = './checkpoints/'
tokenizer_path = './tokenizer/'
feature_extractor_path = './feature_extractor/'


colnames = ["path","labels"]
audio_path = '../input/cleaned-asr-data/data/data/audio/'
transcript_path = '../input/cleaned-asr-data/transcript_durations/dataset_duration_bt_5_and_10.csv'
df = pd.read_csv(transcript_path,usecols=colnames)
df["path"] = audio_path + df["path"] + ".flac"
print(df.shape)
df.head()

TEST_RATIO = 0.90 
VAL_RATIO = 0.75

train_df, test_df = train_test_split(df, random_state = 0, train_size = TEST_RATIO)
train_df, val_df = train_test_split(train_df, random_state = 0, train_size = VAL_RATIO)


train_df.to_csv(csv_folder_path+'/train.csv',index=False)
val_df.to_csv(csv_folder_path+'/val.csv',index=False)
test_df.to_csv(csv_folder_path+'/test.csv',index=False)
!ls {csv_folder_path}


data_files = {
    "train": "train.csv",
    "validation": "val.csv",
    "test": "test.csv"
}

train_data = load_dataset(csv_folder_path, data_files = data_files, split = "train")
val_data = load_dataset(csv_folder_path, data_files = data_files, split = "validation")

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))

show_random_elements(train_data.remove_columns(["path"]), num_examples = 5)

chars_to_ignore = '[\,\?\.\!\-\;\:\"\“\%\‘\”\ ]'

def remove_special_characters(batch):
    batch["labels"] = re.sub(chars_to_ignore, '', batch["labels"]).lower() + " "
    return batch

train_data = train_data.map(remove_special_characters)
val_data = val_data.map(remove_special_characters)


vocab_path = '../input/cleaned-asr-data/data/data/vocabulary/vocab.json'
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_path, 
    unk_token = "[UNK]", 
    pad_token = "[PAD]", 
    word_delimiter_token = "|"
)

tokenizer.save_pretrained(tokenizer_path)

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size = 1, 
    sampling_rate = 16000, 
    padding_value = 0.0, 
    do_normalize = True, 
    return_attention_mask = True
)

processor = Wav2Vec2Processor(
    feature_extractor = feature_extractor, 
    tokenizer = tokenizer
)

processor.save_pretrained(processor_path)

train_data

train_data[0]

val_data[0]

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["labels"]
    
    resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    batch["sampling_rate"] = 16000
    
    return batch

train_data = train_data.map(speech_file_to_array_fn, remove_columns=train_data.column_names)
val_data = val_data.map(speech_file_to_array_fn, remove_columns=val_data.column_names)


rand_int = random.randint(0, len(train_data) - 1)
ipd.Audio(data = np.asarray(
    train_data[rand_int]["speech"]), 
    autoplay = True, 
    rate = 16000
)


rand_int = random.randint(0, len(train_data) - 1)

print("Target text:", train_data[rand_int]["target_text"])
print("Input array shape:", np.asarray(train_data[rand_int]["speech"]).shape)
print("Sampling rate:", train_data[rand_int]["sampling_rate"])



def prepare_dataset(batch, processor):
        
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate = batch["sampling_rate"][0]).input_values
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

train_data = train_data.map(prepare_dataset, fn_kwargs = {"processor": processor}, remove_columns = train_data.column_names, batch_size = 8, num_proc = 4, batched = True)
val_data = val_data.map(prepare_dataset, fn_kwargs = {"processor": processor}, remove_columns = val_data.column_names, batch_size = 8, num_proc = 4, batched = True)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(processor = processor, padding = True)


wer_metric = load_metric("wer")
cer_metric = load_metric("cer", revision = "master")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
    # cer_best = 1 - cer because load best model is considering greater value of cer for better results
    cer_best = 1 - cer
    # wandb.log({'wer':wer, 'cer':cer})
    # experiment.log_metric('wer', wer)
    # experiment.log_metric('cer', cer)
    
    return {"wer": wer, "cer": cer, "cer_best": cer_best}

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", 
    attention_dropout = 0.1,
    hidden_dropout = 0.1,
    feat_proj_dropout = 0.0,
    mask_time_prob = 0.05,
    layerdrop = 0.1,
    gradient_checkpointing = True, 
    ctc_loss_reduction = "mean", 
    pad_token_id = processor.tokenizer.pad_token_id,
    vocab_size = len(processor.tokenizer)
)
# model.config.ctc_zero_infinity = True

model.freeze_feature_encoder()


training_args = TrainingArguments(
  output_dir = checkpoints_path,
  group_by_length=True,
  per_device_train_batch_size=16,
  gradient_accumulation_steps=2,
  evaluation_strategy="steps",
  num_train_epochs=30,
  fp16 = True,
  save_steps = 50,
  eval_steps=50,
  logging_steps=50,
  learning_rate=3e-4,
  warmup_steps=180,
  save_total_limit = 2,
  overwrite_output_dir=True,
  gradient_checkpointing = True,
  metric_for_best_model = 'cer_best',
  load_best_model_at_end=True
)

trainer = Trainer(
    model = model,
    data_collator = data_collator,
    args = training_args,
    compute_metrics = compute_metrics,
    train_dataset = train_data,
    eval_dataset = val_data,
    tokenizer = processor.feature_extractor,
)


trainer.train()

trainer.save_model(trained_model_path)

from IPython.display import FileLink 
!zip -r trained_model.zip {trained_model_path}
!zip -r processor.zip {processor_path}


!ls

FileLink('trained_model.zip')

!zip -r trained_model.zip {trained_model_path}

FileLink(processor.zip)

torch.cuda.empty_cache()

del trainer

!ls ./processor


model = Wav2Vec2ForCTC.from_pretrained("./trained_models/wav2vec0.8dropout").to("cuda")
processor = Wav2Vec2Processor.from_pretrained("./processor/wav2vec0.8dropout")


def segmentLargeArray(inputTensor,chunksize=200000):
    # print(inputTensor)
    list_of_segments = []
    tensor_length = inputTensor.shape[1]
    for i in range(0,tensor_length+1,chunksize):
        list_of_segments.append(inputTensor[:,i:i+chunksize])
    return list_of_segments 


test_audio_ip1 = '/content/gdrive/MyDrive/ASR/ASR/ne_np_female/rec2.wav'
test_audio_ip2 = '/content/gdrive/MyDrive/ASR/ASR/ne_np_female/audio/nep_0258_0119737288.wav'
path = "/content/gdrive/MyDrive/ASR/ASR/ne_np_female/testy.flac"
test1 = '/content/gdrive/MyDrive/ASR/ASR/ne_np_female/test1_anjan.wav'
test2 = '/content/gdrive/MyDrive/ASR/ASR/ne_np_female/rec20sec.wav'
test3 = '/content/gdrive/MyDrive/ASR/ASR/ne_np_female/rec13sec.wav'

test1 = '../input/cleaned-asr-data/data/data/audio/nep_2099_0456476554.flac'
test2 = '../input/cleaned-asr-data/data/data/audio/nep_0546_2868510042.flac'


def predict_from_speech(file):
    speech_array, sampling_rate = torchaudio.load(file)
    # print(speech_array,sampling_rate)
    resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
    resampled_array = resampler(speech_array).squeeze()
    if len(resampled_array.shape) == 1:
        resampled_array = resampled_array.reshape([1,resampled_array.shape[0]])
    # print(resampled_array.shape[1])
    if resampled_array.shape[1] >= 200000:
        print('The input file is longer than 10 seconds')
        list_of_segments = segmentLargeArray(resampled_array)
        # print(list_of_segments)
        output = ''
        for segment in list_of_segments:
            logits = model(segment.to("cuda")).logits
            pred_ids = torch.argmax(logits,dim=-1)[0]
            output += processor.decode(pred_ids)
        print(f"Prediction:\n{output}")
    else:
        print('The input file is less than 10 seconds')
        logits = model(resampled_array.to("cuda")).logits
        # print(logits)
        pred_ids = torch.argmax(logits, dim = -1)[0]
        print("Prediction:")
        print(processor.decode(pred_ids))


test_df = pd.read_csv('../input/cleaned-asr-data/transcript_durations/dataset_duration_gt_10sec.csv')
test_df.head()


test_df['labels'][0]


# predict_from_speech(test_audio_ip1)
predict_from_speech(test1)

test_df['labels'][1]

predict_from_speech(test2)

test3 = f"../input/cleaned-asr-data/data/data/audio/{test_df['path'][2]}.flac"
predict_from_speech(test3)

test_df['labels'][2]

test4 = '../input/cleaned-asr-data/test/test30sec.wav'
predict_from_speech(test4)


test5 = '../input/cleaned-asr-data/test/test1_anjan.wav'
predict_from_speech(test5)



test6 = '../input/cleaned-asr-data/test/rec1.wav'
predict_from_speech(test6)

test7 = '../input/cleaned-asr-data/test/rec2.wav'
predict_from_speech(test7)

test8 = '../input/cleaned-asr-data/test/rec3.wav'
predict_from_speech(test8)


test9 = '../input/cleaned-asr-data/test/rec4.wav'
predict_from_speech(test9)
test10 = '../input/cleaned-asr-data/test/rec5.wav'
predict_from_speech(test10)


!ls ./trained_models/wav2vec0.8dropout


!zip -r processor1.zip ./processor/wav2vec0.8dropout

!mkdir ./trained_models/model_dropout_0.5


trained_model_path = './trained_models/model_dropout_0.5'

