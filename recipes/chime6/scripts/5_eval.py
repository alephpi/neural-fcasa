from pathlib import Path
import pickle
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

def diar2rttm(diar_dir: Path, rttm_dir: Path):

    if (rttm_dir / ".done").exists():
        return

    frame_shift = 0.01 # 10ms frame shift
    diars = list(diar_dir.glob("*.diar"))
    rttm_dir.mkdir(exist_ok=True)


    for diar in tqdm(diars):
        filename = diar.stem
        rttm = filename + ".rttm"

        with open(diar, "rb") as f:
            mask = pickle.load(f)  # shape: [num_speakers, num_frames]
            mask = mask[0,:5,:] # last channel is noise channel

        with open(rttm_dir / rttm, "w") as fout:
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

    (rttm_dir / ".done").touch()
    return

def csv2rttm(src_dir: Path):
    for csv in tqdm(src_dir.glob("*.csv")):
        rttm = csv.with_suffix(".rttm")
        filename = csv.stem
        with open(rttm, "w") as fout:
            with open(csv, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    start_time = float(parts[0])
                    end_time = float(parts[1])
                    spk = parts[2]
                    fout.write(f"SPEAKER {filename} 1 {start_time:.3f} {end_time - start_time:.3f} <NA> <NA> {spk} <NA> <NA>\n")

def count_sca(sys_rttm, ref_rttm):
    from pyannote.database.util import load_rttm

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
        right_num += sys_labels == ref_labels
        sys_res.append(sys_labels)
        ref_res.append(ref_labels)
    return right_num / len(os.listdir(sys_rttm))


def compute_sca(sys_rttm_dir: Path, ref_rttm_dir: Path):
    from pyannote.database.util import load_rttm
    from confidence_intervals import evaluate_with_conf_int

    sys_res = []
    ref_res = []
    sys_rttms = list(sys_rttm_dir.glob("*.rttm"))
    for sys_rttm in tqdm(sys_rttms):
        ref_rttm = ref_rttm_dir / sys_rttm.name
        if len(list(load_rttm(sys_rttm).values())) == 0:
            sys_num_speakers = 0
        else:
            sys_ann = list(load_rttm(sys_rttm).values())[0]
            sys_num_speakers = len(set(sys_ann.labels()))
        if len(list(load_rttm(ref_rttm).values())) == 0:
            ref_num_speakers = 0
        else:
            ref_ann = list(load_rttm(ref_rttm).values())[0]
            ref_num_speakers = len(set(ref_ann.labels()))

        # print(sys_labels, ref_labels)
        sys_res.append(sys_num_speakers)
        ref_res.append(ref_num_speakers)

    def sca_score(sys, ref):
        return np.sum(np.array(sys) == np.array(ref)) / len(sys)

    res = evaluate_with_conf_int(np.array(sys_res), sca_score, np.array(ref_res))
    return res

def compute_der(ref_rttms_dir: Path, sys_rttms_dir: Path):
    """
    Returns:
        dict: each file's DER score and global average DER
    """
    
    from pyannote.core import Timeline, Segment, Annotation
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
    from pyannote.database.util import load_rttm
    # from confidence_intervals import get_bootstrap_indices, get_conf_int
    
    ref_rttms = list(ref_rttms_dir.glob("*.rttm"))
    sys_rttms = list(sys_rttms_dir.glob("*.rttm"))
    if len(ref_rttms) != len(sys_rttms):
        print(f"Warning: reference and system rttm file number mismatch, use a subset of ref rttms, ignore {len(ref_rttms) - len(sys_rttms)} files")
        ref_rttms = [rttm for rttm in ref_rttms if (sys_rttms_dir / rttm.name).exists()]
    
    der_metrics: dict[str, DiarizationErrorRate] = {
                'der_fair_without_overlap': DiarizationErrorRate(collar=0.25, skip_overlap=True), 
                'der_fair': DiarizationErrorRate(collar=0.25, skip_overlap=False),
                'der_full': DiarizationErrorRate(collar=0, skip_overlap=False),
                'der_full_overlap_only': DiarizationErrorRate(collar=0, skip_overlap=False),
           }
        
    jer_metrics: dict[str, JaccardErrorRate] = { 
                'jer_fair_without_overlap': JaccardErrorRate(collar=0.25, skip_overlap=True),
                'jer_fair': JaccardErrorRate(collar=0.25, skip_overlap=False),
                'jer_full': JaccardErrorRate(collar=0, skip_overlap=False),
                'jer_full_overlap_only': JaccardErrorRate(collar=0, skip_overlap=False),
                }
    
    der_metrics_result = {key: [np.nan, np.nan, np.nan, np.nan] for key in der_metrics.keys()}
    der_metrics_results = {key: [] for key in der_metrics.keys()}
    jer_metrics_result = {key: np.nan for key in jer_metrics.keys()}
    jer_metrics_results = {key: [] for key in jer_metrics.keys()}

    def detailed(der) -> list[float]:
        if der['total'] == 0: # possible when measuring der fair without overlap
            return [0, 0, 0, 0]
        miss = der['missed detection'] / der['total']
        fa = der['false alarm'] / der['total']
        confusion = der['confusion'] / der['total']
        der_ = der['diarization error rate']
        return [miss, fa, confusion, der_]

    for ref_rttm in tqdm(ref_rttms):
        uri = ref_rttm.stem
        sys_rttm = sys_rttms_dir / ref_rttm.name
        reference: Annotation = load_rttm(ref_rttm).get(uri, None)
        hypothesis: Annotation = load_rttm(sys_rttm).get(uri, None)
        if not reference:
            # if reference contains no speech, skip it
            continue
        else:
            uem = Timeline([Segment(0, 10)], uri=uri)
            uem_overlap_only = reference.get_overlap()
            if not hypothesis:
                # always miss, der == 1
                for key in der_metrics.keys():
                    der_metrics_result[key] = [1,0,0,1]
            else:
                for key in der_metrics.keys():
                    der_metric = der_metrics[key]
                    if key.endswith('overlap_only') and uem_overlap_only:
                        der_metrics_result[key] = detailed(der_metric(reference, hypothesis, uem=uem_overlap_only, detailed=True))
                    else:
                        der_metrics_result[key] = detailed(der_metric(reference, hypothesis, uem=uem, detailed=True))

                    der_metrics_results[key].append(der_metrics_result[key])

                for key in jer_metrics.keys():
                    jer_metric = jer_metrics[key]
                    if key.endswith('overlap_only') and uem_overlap_only:
                        jer_metrics_result[key] = jer_metric(reference, hypothesis, uem=uem_overlap_only, detailed=False) # type: ignore
                    else:
                        try:
                            jer_metrics_result[key] = jer_metric(reference, hypothesis, uem=uem, detailed=False) # type: ignore
                        except ZeroDivisionError:
                            jer_metrics_result[key] = np.nan

                    jer_metrics_results[key].append(jer_metrics_result[key])

    # der_metrics_results = {key: np.array(value) for key, value in der_metrics_results.items()}
    der_metrics_results_mean = {key: np.nanmean(value, 0) for key, value in der_metrics_results.items()}
    jer_metrics_results_mean = {key: np.nanmean(value, 0) for key, value in jer_metrics_results.items()}
    return der_metrics_results_mean, jer_metrics_results_mean

    # der_bootstrapped = []
    # num_samples = len(ders_full)
    # num_bootstraps = 1000 # int(50/alpha*100), where alpha = 5
    # for nb in np.arange(num_bootstraps):
    #     indices = get_bootstrap_indices(num_samples, None, random_state=nb)
    #     der_bootstrapped.append(ders_full[indices])
    # compute global average DER
    # der_conf_int = get_conf_int(der_bootstrapped, alpha=5)

    # return der_mean, der_conf_int

if __name__ == "__main__":
    # import debugpy
    # try:
    #     debugpy.listen(('localhost', 9504))
    #     print('Waiting for debugger attach')
    #     debugpy.wait_for_client()
    # except Exception as e:
    #     pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_rttm_dir", type=str, default="/home/ids/smao-22/phd/neural-fcasa/recipes/chime6/chime6corpus/processed_data/tt/rttm",help="reference label rttm")
    parser.add_argument("--sys_rttm_dir", type=str, help="system rttm directory")
    parser.add_argument("--diar_dir", type=str, help="system diar directory")
    args = parser.parse_args()

    ref_rttm_dir = Path(args.ref_rttm_dir)
    assert args.diar_dir or args.sys_rttm_dir, "either diar_dir or sys_rttm_dir should be provided"
    if args.diar_dir:
        diar_dir = Path(args.diar_dir)
        sys_rttm_dir = diar_dir.parent / 'rttm'
        diar2rttm(diar_dir, sys_rttm_dir)
    else:
        sys_rttm_dir = Path(args.sys_rttm_dir)

    # print("---sca---")
    # sca, (sca_lower, sca_upper) = compute_sca(sys_rttm_dir, ref_rttm_dir)
    # print(sca, sca_lower-sca, sca_upper-sca)

    der, jer = compute_der(ref_rttm_dir, sys_rttm_dir)
    print("---der---")
    der = pd.DataFrame.from_dict(der, orient='index', columns=['miss', 'fa', 'confusion', 'der'])
    der.index.name = "metric"
    print(der.to_csv())
    print("---jer---")
    jer = pd.DataFrame.from_dict(jer, orient='index', columns=['jer'])
    jer.index.name = "metric"
    print(jer.to_csv())

    # der, (der_lower, der_upper) = compute_der(ref_rttm_dir, sys_rttm_dir)
    # print(der, der_lower, der_upper)