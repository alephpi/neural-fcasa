import torch
import numpy as np
from datasets import load_from_disk

class WavActTransform:
    def __init__(
        self,
        duration: int | None,
        sr: int,
        hop_length: int,
        randperm_mic: bool = True,
        randperm_spk: bool = True,
    ):
        self.duration = duration
        self.sr = sr
        self.hop_length = hop_length
        self.randperm_mic = randperm_mic
        self.randperm_spk = randperm_spk
        self.duration_frame = self.sr * self.duration // self.hop_length if self.duration is not None else None
    
    def __call__(self, batch):
        batch_wav = torch.tensor(batch["wav"])
        batch_act = torch.tensor(batch["act"])
        transformed_wav = []
        transformed_act = []

        for wav, act in zip(batch_wav, batch_act):
            if self.duration_frame is not None:
                t_start_act = np.random.randint(0, act.shape[1] - self.duration_frame + 1)
                t_end_act = t_start_act + self.duration_frame
                act = act[:, t_start_act:t_end_act]
                
                t_start = self.hop_length * t_start_act
                t_end = self.hop_length * t_end_act
                wav = wav[:, t_start:t_end]
            
            if self.randperm_mic:
                wav = wav[torch.randperm(wav.shape[0])]
            
            if self.randperm_spk:
                act = act[torch.randperm(act.shape[0])]

            transformed_wav.append(wav)
            transformed_act.append(act)

        batch_wav = torch.stack(transformed_wav)
        batch_act = torch.stack(transformed_act)

        return {"wav": batch_wav, "act": batch_act}

class WavActDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path,
        duration: int | None,
        sr: int,
        hop_length: int,
        randperm_mic: bool = True,
        randperm_spk: bool = True,
        keep_in_memory: bool = False,
        ):

        self._dataset = load_from_disk(dataset_path, keep_in_memory)
        self._dataset.set_format('torch')
        self.duration = duration
        self.sr = sr
        self.hop_length = hop_length

        self.randperm_mic = randperm_mic
        self.randperm_spk = randperm_spk
        self.duration_frame = self.sr * self.duration // self.hop_length if self.duration is not None else None
    
    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        item = self._dataset[index]
        wav: torch.Tensor = item['wav']
        act: torch.Tensor = item['act']
        if self.duration_frame is not None:
            t_start_act = np.random.randint(0, act.shape[1] - self.duration_frame + 1)
            t_end_act = t_start_act + self.duration_frame
            act = act[:, t_start_act:t_end_act]
            
            t_start = self.hop_length * t_start_act
            t_end = self.hop_length * t_end_act
            wav = wav[:, t_start:t_end]
        
        if self.randperm_mic:
            wav = wav[torch.randperm(wav.shape[0])]
        
        if self.randperm_spk:
            act = act[torch.randperm(act.shape[0])]

        return {'wav': wav, 'act': act}
