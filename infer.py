from huggingface_hub import hf_hub_download
import os, zipfile
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset, Audio
import soundfile as sf
import torch
import kenlm
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from pvrecorder import PvRecorder
import wave 
import struct
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

# wav
class Wav2Vec2():
    def __init__(self, cache_path, wav2vec2_path):
        self.cache_path = cache_path
        self.wav2vec2_path = wav2vec2_path

    def get_processor(self):
        processor = Wav2Vec2Processor.from_pretrained(self.wav2vec2_path, cache_dir=self.cache_path)
        self.processor = processor

    def get_model(self):
        model = Wav2Vec2ForCTC.from_pretrained(self.wav2vec2_path, cache_dir=self.cache_path)
        self.model = model
    
    def get_lm_file(self):

        lm_file = hf_hub_download(repo_id=self.wav2vec2_path, filename="vi_lm_4grams.bin.zip", cache_dir=self.cache_path)
        print(lm_file)
        with zipfile.ZipFile(lm_file, 'r') as zip_ref:
            zip_ref.extractall(self.cache_path)

        lm_file = self.cache_path + 'vi_lm_4grams.bin'

        self.lm_file = lm_file
    
    def get_decoder_ngram_model(self):
        vocab_dict = self.processor.tokenizer.get_vocab()
        sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
        vocab = [x[1] for x in sort_vocab][:-2]
        vocab_list = vocab

        # convert ctc blank character representation
        vocab_list[self.processor.tokenizer.pad_token_id] = ""

        # replace special characters
        vocab_list[self.processor.tokenizer.unk_token_id] = ""

        # convert space character representation
        vocab_list[self.processor.tokenizer.word_delimiter_token_id] = " "

        # specify ctc blank char index, since conventially it is the last entry of the logit matrix
        alphabet = Alphabet.build_alphabet(
            vocab_list, ctc_token_idx=self.processor.tokenizer.pad_token_id)
        lm_model = kenlm.Model(self.lm_file)
        decoder = BeamSearchDecoderCTC(alphabet,
                                    language_model=LanguageModel(lm_model))
        self.ngram_lm_model = decoder
        # return self.ngram_lm_model
    def get_device(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

# load wav
wav2vec2 = Wav2Vec2(cache_path="./cache/", wav2vec2_path='nguyenvulebinh/wav2vec2-base-vietnamese-250h')
wav2vec2.get_processor()
wav2vec2.get_model()
wav2vec2.get_lm_file()
wav2vec2.get_decoder_ngram_model()
wav2vec2.get_device()

def add_column_w2v2_transcription(example):
    audio = example["audio"]
    input_values = wav2vec2.processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_values.to(wav2vec2.device)
    with torch.no_grad():
        wav2vec2.model.to(wav2vec2.device)
        logits = wav2vec2.model(input_values).logits[0]
        transcription = wav2vec2.ngram_lm_model.decode(logits.cpu().detach().numpy(), beam_width=500)

    # Add w2v2 transcription
    example["w2v2_baseline_transcription"] = transcription

    # Empty cuda
    del input_values
    del logits
    torch.cuda.empty_cache()

    # Return the modified example
    return example

# record
recorder = PvRecorder(device_index=-1, frame_length=512)
audio = []
try:
    recorder.start()
    while True:
        frame = recorder.read()
        audio.extend(frame)
except KeyboardInterrupt:
    recorder.stop()
    with wave.open('record.wav', 'w') as f:
        f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
        f.writeframes(struct.pack("h" * len(audio), *audio))
finally:
    recorder.delete()

ds = load_dataset("audio")

result = ds["train"].map(add_column_w2v2_transcription)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
new_model = AutoModelForSequenceClassification.from_pretrained("linhtran92/intentdetection")
new_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

import numpy as np

def get_prediction(text):
    encoding = new_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    #encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}

    outputs = new_model(**encoding)

    logits = outputs.logits
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    softmax = torch.nn.Softmax()
    print(softmax)
    probs = softmax(logits.squeeze().cpu())
    probs = probs.detach().numpy()
    label = np.argmax(probs, axis=-1)

    if label == 0:
        return {
            'tag': 'hỏa hoạn',
            'probability': [probs[0], probs[1], probs[2]]
        }
    elif label == 1:
        return {
            'tag': 'đi lạc',
            'probability': [probs[0], probs[1], probs[2]]
        }
    else:
        return {
            'tag': 'mắc kẹt',
            'probability': [probs[0], probs[1], probs[2]]
        }

tag = get_prediction(result[0]['w2v2_baseline_transcription'])
print(tag)