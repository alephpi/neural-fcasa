import pickle
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse


def vad(src_path:str, dst_path:str):
    from pyannote.audio import Pipeline
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",use_auth_token="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")
    pipeline.segmentation = "pyannote/segmentation-3.0.0"
    pipeline.to(device)
    # src_path / separated_dir / wav_file
    for idx in tqdm(os.listdir(src_path)):
        print(idx)
        output_path = os.path.join(dst_path, f"{idx}.rttm")
        annotations = []
        for i in range(5):
            wav_path = os.path.join(src_path, idx, f"{i}.wav")
            output = pipeline(wav_path)
            output.uri = idx
            for segment, track, label in output.itertracks(yield_label=True):
                if label == "SPEECH":
                    output[segment, track] = i
            annotations.append(output)
        with open(output_path, "w") as f:
            for ann in annotations:
                ann.write_rttm(f)
            
        # break
    return

def separate_channel(src_path:str, dst_path:str):
    """
    separate the channel of the audio file, and save the separated audio files to the dst_path
    """
    import soundfile as sf
    for file in tqdm(os.listdir(src_path)):
        if not file.endswith(".wav"):
            continue
        tempdir = os.path.join(dst_path, file.replace(".wav", ""))
        os.makedirs(tempdir, exist_ok=True)
        file_path = os.path.join(src_path, file)
        audio, sr = sf.read(file_path)
        print(audio.shape)
        for i in range(audio.shape[1]-1):
            sf.write(os.path.join(tempdir, f"{i}.wav"), audio[:,i], sr)
    return

def count_sca(sys_rttm, ref_rttm):
    from pyannote.database.util import load_rttm
    from pyannote.core import Annotation

    right_num = 0
    for rttm in tqdm(os.listdir(sys_rttm)):
        if not rttm.endswith(".rttm"):
            continue
    
        sys_rttm_path = os.path.join(sys_rttm, rttm)
        ref_rttm_path = os.path.join(ref_rttm, rttm)
        
        if len(list(load_rttm(sys_rttm_path).values())) == 0:
            sys_labels = 0
        else:
            sys_ann = list(load_rttm(sys_rttm_path).values())[0]
            sys_labels = len(set(sys_ann.labels()))            
        if len(list(load_rttm(ref_rttm_path).values())) == 0:
            ref_labels = 0
        else:
            ref_ann = list(load_rttm(ref_rttm_path).values())[0]
            ref_labels = len(set(ref_ann.labels()))
        

        # print(sys_labels, ref_labels)
        right_num += sys_labels == ref_labels
    return right_num / len(os.listdir(sys_rttm))


def count_sca_with_conf_int(sys_rttm, ref_rttm):
    from pyannote.database.util import load_rttm
    from pyannote.core import Annotation
    from confidence_intervals import evaluate_with_conf_int
    
    right_num = 0
    sys_res = []
    ref_res = []
    for rttm in tqdm(os.listdir(sys_rttm)):
        if not rttm.endswith(".rttm"):
            continue
        sys_rttm_path = os.path.join(sys_rttm, rttm)
        ref_rttm_path = os.path.join(ref_rttm, rttm)
        
        if len(list(load_rttm(sys_rttm_path).values())) == 0:
            sys_labels = 0
        else:
            sys_ann = list(load_rttm(sys_rttm_path).values())[0]
            sys_labels = len(set(sys_ann.labels()))            
        if len(list(load_rttm(ref_rttm_path).values())) == 0:
            ref_labels = 0
        else:
            ref_ann = list(load_rttm(ref_rttm_path).values())[0]
            ref_labels = len(set(ref_ann.labels()))
        

        # print(sys_labels, ref_labels)
        right_num += sys_labels == ref_labels
        sys_res.append(sys_labels)
        ref_res.append(ref_labels)
        
        def sca_score(sys, ref):
            return np.sum(np.array(sys) == np.array(ref)) / len(sys)

    res = evaluate_with_conf_int(np.array(sys_res), sca_score, np.array(ref_res))
    print(res)

    return res

def batch_compute_der(reference_rttm_list, hypothesis_rttm_list):
    """
    
    Args:
        reference_rttm_list (list): reference RTTM file path list
        hypothesis_rttm_list (list): system output RTTM file path list
        
    Returns:
        dict: each file's DER score and global average DER
    """
    
    from pyannote.metrics.diarization import DiarizationErrorRate
    from pyannote.core import Annotation
    from pyannote.database.util import load_rttm, load_uem
    
    if len(reference_rttm_list) != len(hypothesis_rttm_list):
        raise ValueError("reference and system rttm file number mismatch")
    metric = DiarizationErrorRate()
    results = {}
    
    for ref_path, hyp_path in tqdm(zip(reference_rttm_list, hypothesis_rttm_list)):
        try:
            # get file name(without extension) as URI
            uri = os.path.splitext(os.path.basename(ref_path))[0]
            
            # load RTTM file
            reference = load_rttm(ref_path)[uri]
            hypothesis = load_rttm(hyp_path)[uri]
            
            # computeDER
            der = metric(reference, hypothesis, detailed=False)
            
            results[uri] = der
            # print(f"{uri}: DER = {der:.3f}")
            
        except Exception as e:
            print(f"Error processing file pair {ref_path} and {hyp_path}: {str(e)}")
            results[uri] = None
    
    # compute global average DER
    valid_scores = [v for v in results.values() if v is not None]
    global_der = sum(valid_scores) / len(valid_scores) if valid_scores else None
    
    return {
        "file_scores": results,
        "global_der": global_der
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_rttm", type=str, default="/home/ids/bli-24/data/ami/ref",help="reference label rttm")
    parser.add_argument("--sys_wav", type=str, default="/home/ids/bli-24/data/ami/our_base",help="output wavefile directory")
    parser.add_argument("--save_dir", type=str, default="/home/ids/bli-24/data/baseline",help="save middle results directory")
    parser.add_argument("--separate", type=bool, default=False,help="whether to separate channel")
    parser.add_argument("--vad", type=bool, default=False,help="whether to vad")
    parser.add_argument("--sca", type=bool, default=False,help="whether to compute sca")
    parser.add_argument("--der", type=bool, default=True,help="whether to compute der")
    args = parser.parse_args()
    
    
    # First step: vad sys output to rttm
    separate_dir = os.path.join(args.save_dir, "separate")
    if args.separate:
        os.makedirs(separate_dir, exist_ok=True)
        print("---separate channel---")
        separate_channel(args.sys_wav, separate_dir)
    
    vad_dir = os.path.join(args.save_dir, "vad")
    if args.vad:
        os.makedirs(vad_dir, exist_ok=True)
        print("---vad---")
        vad(separate_dir, vad_dir)
    
    # Second step: compute sca
    if args.sca:
        print("---sca---")
        sca = count_sca_with_conf_int(vad_dir, args.ref_rttm)
        print(sca)
    
    # Third step: compute der
    if args.der:    
        print("---der---")
        ref_rttm_list = [os.path.join(args.ref_rttm, rttm) for rttm in os.listdir(args.ref_rttm)]
        sys_rttm_list = [os.path.join(vad_dir, rttm) for rttm in os.listdir(args.ref_rttm)]
        der = batch_compute_der_with_conf_int(ref_rttm_list, sys_rttm_list)
        print(der)


    
    
    
    
    
    
    
    
    
    
    

    