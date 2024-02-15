import os
import scipy.io
from pathlib import Path
import python_speech_features as psf
import numpy as np
import random
from typing import Literal
import torch

class SEP_Dataset(torch.utils.data.Dataset):
    SRC_DIR = "/home/priyanka/Desktop/MS_FINAL/SEP28K"
    FS = 16000
    AUDIO_DUR = 3
    WIN_LEN = 320 # 20 ms
    HOP_LEN = 160 # 10 ms
    LABEL_TYPE = Literal["Interjection", "Prolongation", "WordRep", "Fluent", "SoundRep"]

    def __init__(self, class_a: LABEL_TYPE, class_b: LABEL_TYPE, do_shuffle: bool = True):
        src_path_a = os.path.join(self.SRC_DIR, class_a)
        src_path_b = os.path.join(self.SRC_DIR, class_b)
        data_a = list(Path(src_path_a).glob("*.wav"))
        data_b = list(Path(src_path_b).glob("*.wav"))
        print(f"Found {len(data_a)} samples in class_a: {class_a} ({src_path_a})")
        print(f"Found {len(data_b)} samples in class_b: {class_b} ({src_path_b})")
        print(f"Total samples: {len(data_a) + len(data_b)}")
        if do_shuffle:
            random.shuffle(data_a)
            random.shuffle(data_b)
        data_a = data_a[: min(len(data_a), len(data_b))]
        data_b = data_b[: len(data_a)]
        self.data = [(sample, 0) for sample in data_a] + [(sample, 1) for sample in data_b]
        print(f"After balancing, total samples: {len(self.data)} (each class: {len(data_a)})")
        if do_shuffle:
            random.shuffle(self.data)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        wav_path, label = self.data[ind]
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
    dataset = SEP_Dataset(
        class_a="SoundRep",
        class_b="Fluent"
    )
    print(len(dataset))
    feats, label = dataset[0]
    print(feats.shape, label)
