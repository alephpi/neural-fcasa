import argparse
import multiprocessing as mp
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


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
    from confidence_intervals import evaluate_with_conf_int
    from pyannote.database.util import load_rttm

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

def _detailed(der) -> list[float]:
    if der['total'] == 0: # possible when measuring der fair without overlap
        return [np.nan] * 4
    miss = der['missed detection'] / der['total']
    fa = der['false alarm'] / der['total']
    confusion = der['confusion'] / der['total']
    der_ = der['diarization error rate']
    return [miss, fa, confusion, der_]


def _eval_one(args: tuple[Path, Path]):
    """Compute DER/JER of a single file. Returns None if the reference has no speech."""
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.database.util import load_rttm
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

    ref_rttm, sys_rttm = args

    der_metrics: dict[str, DiarizationErrorRate] = {
                'forgive': DiarizationErrorRate(collar=0.25, skip_overlap=True), 
                'fair': DiarizationErrorRate(collar=0.25, skip_overlap=False),
                'full': DiarizationErrorRate(collar=0, skip_overlap=False),
                'overlap': DiarizationErrorRate(collar=0, skip_overlap=False),
           }
        
    jer_metrics: dict[str, JaccardErrorRate] = { 
                'forgive': JaccardErrorRate(collar=0.25, skip_overlap=True),
                'fair': JaccardErrorRate(collar=0.25, skip_overlap=False),
                'full': JaccardErrorRate(collar=0, skip_overlap=False),
                'overlap': JaccardErrorRate(collar=0, skip_overlap=False),
                }

    uri = ref_rttm.stem
    reference: Annotation = load_rttm(ref_rttm).get(uri, None)
    hypothesis: Annotation = load_rttm(sys_rttm).get(uri, None)
    if not reference:
        # if reference contains no speech, skip it
        return None
    uem = Timeline([Segment(0, 10)], uri=uri)
    uem_overlap_only = reference.get_overlap()
    empty_hyp = not hypothesis
    if empty_hyp:
        hypothesis = Annotation(uri=uri)

    der_res: dict[str, list[float]] = {}
    for key, der_metric in der_metrics.items():
        if key == 'overlap':
            if not uem_overlap_only:
                continue # no overlap in this file, metric undefined
            key_uem = uem_overlap_only
        else:
            key_uem = uem
        der_res[key] = _detailed(der_metric(reference, hypothesis, uem=key_uem, detailed=True))

    jer_res: dict[str, float] = {}
    for key, jer_metric in jer_metrics.items():
        if key == 'overlap':
            if not uem_overlap_only:
                continue
            key_uem = uem_overlap_only
        else:
            key_uem = uem
        if empty_hyp:
            jer_res[key] = 1.0 # every reference speaker is missed
            continue
        try:
            jer_res[key] = jer_metric(reference, hypothesis, uem=key_uem, detailed=False) # type: ignore
        except ZeroDivisionError:
            jer_res[key] = np.nan

    return der_res, jer_res


def compute_der(ref_rttms_dir: Path, sys_rttms_dir: Path, num_workers: int = 8):
    """
    Returns:
        dict: each file's DER score and global average DER
    """
    ref_rttms = list(ref_rttms_dir.glob("*.rttm"))
    sys_rttms = list(sys_rttms_dir.glob("*.rttm"))
    if len(ref_rttms) != len(sys_rttms):
        raise ValueError("reference and system rttm file number mismatch")

    tasks = [(ref_rttm, sys_rttms_dir / ref_rttm.name) for ref_rttm in ref_rttms]
    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(_eval_one, tasks, chunksize=4), total=len(tasks)))
    else:
        results = [_eval_one(task) for task in tqdm(tasks)]

    der_keys = ['forgive', 'fair', 'full', 'overlap']
    jer_keys = ['forgive', 'fair', 'full', 'overlap']
    der_metrics_results = {key: [] for key in der_keys}
    jer_metrics_results = {key: [] for key in jer_keys}
    for res in results:
        if res is None:
            continue
        for key, value in res[0].items():
            der_metrics_results[key].append(value)
        for key, value in res[1].items():
            jer_metrics_results[key].append(value)

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
    parser.add_argument("--num_workers", type=int, default=16, help="number of worker processes for DER/JER")
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

    der, jer = compute_der(ref_rttm_dir, sys_rttm_dir, args.num_workers)
    print("---result---")
    der_df: pd.DataFrame = pd.DataFrame.from_dict(der, orient='index', columns=['miss', 'fa', 'conf', 'der'])
    der_df.index.name = "metric"
    jer_df = pd.DataFrame.from_dict(jer, orient='index', columns=['jer'])
    jer_df.index.name = "metric"
    df = pd.DataFrame.join(der_df, jer_df, how='inner')
    df *= 100
    print(df.to_csv(sep="\t", float_format="%.2f"))