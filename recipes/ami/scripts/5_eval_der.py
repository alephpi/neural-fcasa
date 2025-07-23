import pickle
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

def diar2rttm(src_dir, dst_dir):

    frame_shift = 0.01 # 10ms frame shift
    
    os.makedirs(dst_dir, exist_ok=True)
    
    for diar in tqdm(os.listdir(src_dir)):
        if not diar.endswith(".diar"):
            continue
        diar_path = os.path.join(src_dir, diar)
        # print(diar_path)
        output_rttm = os.path.join(dst_dir, diar.replace(".diar", ".rttm"))
        
        filename = os.path.splitext(diar)[0]
    
        with open(diar_path, "rb") as f:
            mask = pickle.load(f)  # shape: [num_speakers, num_frames]
            mask = mask[0,:5,:] # last channel is noise channel

        with open(output_rttm, "w") as fout:
            num_speakers, num_frames = mask.shape
            for spk in range(num_speakers):
                active = mask[spk]
                in_segment = False
                start_time = 0
                for i, val in enumerate(active):
                    if val > 0 and not in_segment:
                        start_time = i * frame_shift
                        in_segment = True
                    elif val == 0 and in_segment:
                        end_time = i * frame_shift
                        fout.write(f"SPEAKER {filename} 1 {start_time:.3f} {end_time - start_time:.3f} <NA> <NA> {spk} <NA> <NA>\n")
                        in_segment = False

                if in_segment:
                    end_time = num_frames * frame_shift
                    fout.write(f"SPEAKER {filename} 1 {start_time:.3f} {end_time - start_time:.3f} <NA> <NA> {spk} <NA> <NA>\n")


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
    
    parser.add_argument("--der", type=bool, default=True,help="whether to compute der")
    args = parser.parse_args()
    
    
    if args.der:    
        print("---der---")
        save_dir = os.path.join(args.save_dir, "diar2rttm")
        diar2rttm(args.sys_wav, save_dir)
        
        ref_rttm_list = [os.path.join(args.ref_rttm, rttm) for rttm in os.listdir(args.ref_rttm)]
        sys_rttm_list = [os.path.join(save_dir, rttm) for rttm in os.listdir(args.ref_rttm)]
        der = batch_compute_der(ref_rttm_list, sys_rttm_list)
        print(der)


    
    
    
    
    
    
    
    
    
    
    

    