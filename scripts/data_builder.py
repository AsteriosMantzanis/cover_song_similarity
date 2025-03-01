import argparse
import os

import h5py
import pandas as pd
from loguru import logger as logging
from tqdm import tqdm


def process_h5_files(input_dir, output_csv) -> pd.DataFrame:
    """Extracts HPCP & Chroma features from HDF5 files and saves to CSV."""

    data = []

    for i in tqdm(os.listdir(input_dir), desc="Processing folders"):
        work = i.split("_")[-1]
        folder = os.path.isdir(os.path.join(input_dir, i))  # Ensure it's a directory

        if folder:
            inner_folder = os.path.join(input_dir, i)

            for j in os.listdir(inner_folder):
                if j.endswith(".h5"):
                    with h5py.File(os.path.join(inner_folder, j), "r") as f:
                        performance = j.split(".")[0].split("_")[-1]

                        hpcp = f["hpcp"][:].mean(axis=0)
                        chroma = f["chroma_cens"][:].mean(axis=0)

                        data.append(
                            {
                                "work": work,
                                "performance": performance,
                                **{f"hpcp_{i}": hpcp[i] for i in range(12)},
                                **{f"chroma_{i}": chroma[i] for i in range(12)},
                            }
                        )

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    logging.info(f"CSV saved: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract HPCP & Chroma features from HDF5 files."
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
