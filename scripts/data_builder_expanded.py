import argparse
import os

import h5py
import numpy as np
import pandas as pd
from loguru import logger as logging
from tqdm import tqdm


def process_h5_files(input_dir, output_csv) -> pd.DataFrame:
    """Extracts HPCP, Chroma, MFCC, Crema, Novelty functions, and Tempo from HDF5 files and saves to CSV."""

    data = []

    for i in tqdm(os.listdir(input_dir), desc="Processing works"):
        work = i.split("_")[-1]
        folder = os.path.isdir(os.path.join(input_dir, i))  # Ensure it's a directory

        if folder:
            inner_folder = os.path.join(input_dir, i)

            for j in os.listdir(inner_folder):
                if j.endswith(".h5"):
                    with h5py.File(os.path.join(inner_folder, j), "r") as f:
                        performance = j.split(".")[0].split("_")[-1]

                        # Extract features
                        hpcp = f["hpcp"][:].mean(axis=0)
                        chroma = f["chroma_cens"][:].mean(axis=0)
                        mfcc_htk = (
                            np.ma.masked_invalid(f["mfcc_htk"][:]).mean(axis=1).data
                        )
                        crema = f["crema"][:].mean(axis=0)

                        # Extract Madmom features (novelty & tempo)
                        madmom_feats = f["/madmom_features"]
                        novfn = madmom_feats["novfn"][:].mean().item()
                        snovfn = madmom_feats["snovfn"][:].mean().item()
                        tempo = madmom_feats["tempos"][0, 0].item()

                        # Append to data list
                        data.append(
                            {
                                "work": work,
                                "performance": performance,
                                **{f"hpcp_{i}": hpcp[i] for i in range(12)},
                                **{f"chroma_{i}": chroma[i] for i in range(12)},
                                **{
                                    f"mfcc_{i}": mfcc_htk[i] for i in range(13)
                                },  # MFCC has 13 dimensions
                                **{f"crema_{i}": crema[i] for i in range(12)},
                                "novfn": novfn,
                                "snovfn": snovfn,
                                "tempo": tempo,
                            }
                        )

    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    logging.info(f"CSV saved: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract musical features from Da-TACOS dataset."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Path to the dataset folder."
    )
    parser.add_argument(
        "--output_csv", type=str, required=True, help="Path to save the output CSV."
    )

    args = parser.parse_args()
    process_h5_files(args.input_dir, args.output_csv)


if __name__ == "__main__":
    main()
