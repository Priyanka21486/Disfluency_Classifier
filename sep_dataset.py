import os
import scipy.io
from pathlib import Path
import python_speech_features as psf
import numpy as np
import random
from typing import Literal
import torch

class SEP_Binary_Dataset(torch.utils.data.Dataset):
    SRC_DIR = "/home/priyanka/Desktop/MS_FINAL/SSP/SEP28K"
    FS = 16000
    AUDIO_DUR = 3
    WIN_LEN = 320 # 20 ms
    HOP_LEN = 160 # 10 ms
    DISFLUENCY_TYPE = Literal["Interjection", "Prolongation", "WordRep"]

    def __init__(self, disfluency_type: DISFLUENCY_TYPE):
        self.disfluency_path = os.path.join(self.SRC_DIR, disfluency_type)
        self.fluent_path = os.path.join(self.SRC_DIR, "Fluent")
        self.disfluency_data = list(Path(self.disfluency_path).glob("*.wav"))
        self.fluent_data = list(Path(self.fluent_path).glob("*.wav"))
        print(f"Found {len(self.disfluency_data)} samples in disfluency directory ({self.disfluency_path})")
        print(f"Found {len(self.fluent_data)} samples in fluent directory ({self.fluent_path})")
        self._balance_data()
        self.total_data = [(sample, 1) for sample in self.disfluency_data]
        self.total_data += [(sample, 0) for sample in self.fluent_data]
        # random.shuffle(self.total_data)
    
    def _balance_data(self):
        min_len = min(len(self.disfluency_data), len(self.fluent_data))
        self.disfluency_data = random.sample(self.disfluency_data, k=min_len)
        self.fluent_data = random.sample(self.fluent_data, k=min_len)
        print(f"Balancing data -> disfluency: {len(self.disfluency_data)}, fluent: {len(self.fluent_data)}")

    def __len__(self):
        return len(self.total_data)

    def __getitem__(self, ind):
        wav_path, label = self.total_data[ind]
        wav = self._load_audio(wav_path)
        feats = self._extract_features(wav)
        return feats, label

    def _load_audio(self, path):
        fs, wav = scipy.io.wavfile.read(path)
        assert fs == self.FS
        assert wav.shape[0] == self.AUDIO_DUR * self.FS, f"invalid audio length for path: {path}"
        return wav
    
    def _extract_features(self, wav):
        return psf.mfcc(
            wav,
            samplerate=self.FS,
            winlen=(self.WIN_LEN / self.FS),
            winstep=(self.HOP_LEN / self.FS),
            nfft=self.WIN_LEN,
            nfilt=26,
            numcep=13
        )
    
    @staticmethod
    def collate_fn(batch):
        n_frames, n_feats = batch[0][0].shape
        feats = torch.FloatTensor(len(batch), n_frames, n_feats)
        labels = torch.FloatTensor(len(batch))
        for ind, (feat, label) in enumerate(batch):
            feats[ind, :, :] = torch.FloatTensor(feat)
            labels[ind] = label
        return feats, labels

if __name__ == "__main__":
    dataset = SEP_Binary_Dataset(disfluency_type="WordRep")
    print(len(dataset))
    feats, label = dataset[0]
    print(feats.shape, label)
