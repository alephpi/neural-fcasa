from pathlib import Path
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import numpy as np

path = Path('/home/ids/smao-22/phd/neural-fcasa/recipes/ali/alicorpus/processed_data/tr/derev')

def check_file(file_path):
    try:
        wav, _ = sf.read(file_path)
        corr_matrix = np.corrcoef(wav[:, 0], wav[:, 1])
        if corr_matrix[0,1] == 1.0:
            return str(file_path)
    except Exception:
        return str(file_path) + ' (error)'
    return None

def main():
    files = list(path.glob('*.wav'))
    results = []
    # limit workers to avoid too many open files
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = {ex.submit(check_file, str(f)): f for f in files}
        with tqdm(total=len(futures), desc='Checking files') as pbar:
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    results.append(res)
                pbar.update(1)

    for r in results:
        # keep original messaging for errors too
        if r.endswith(' (error)'):
            print(f"{r}")
        else:
            print(f"{r} has correlation of 1.0.")

if __name__ == '__main__':
    main()