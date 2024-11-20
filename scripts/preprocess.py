import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import gzip
import shutil
import requests
from tqdm import tqdm
import csv
import os

FILE_PATH = "data/uniprot_sprot.csv"

def parse_uniprot_flatfile(file_path):
    # Function to parse the UniProt flat file
    # ID (Identifier): Unique protein entry name, includes protein name and species (e.g., A4_HUMAN).
    # AC (Accession Number): Unique ID for the protein in the database (e.g., P12345;).
    # DE (Description): Official and alternative names of the protein (e.g., Full=Alpha-4 integrin;).
    # GN (Gene Name): Gene encoding the protein, with synonyms (e.g., Name=ITGA4; Synonyms=CD49D;).
    # OS (Organism Source): Organism where the protein originates (e.g., Homo sapiens (Human)).
    # OC (Organism Classification): Taxonomic classification of the organism (e.g., Eukaryota; Mammalia; Homo.).
    # SQ (Sequence): Amino acid sequence, length, molecular weight, and checksum (e.g., 267 AA; 29178 MW; MALWMR...).
    # DR (Cross-References): Links to related databases like PDB, Ensembl, or OMIM (e.g., PDB; 1A4Y;).
    # CC (Comments): Functional or disease-related notes (e.g., -!- FUNCTION: Cell adhesion.).
    # KW (Keywords): Tags summarizing protein features (e.g., Cell adhesion; Disease mutation.).
    # FT (Feature Table): Specific features, like binding sites or modification regions (e.g., REGION 1..50 Signal peptide.).
    data = []
    entry = {}
    with open(file_path, 'r') as f:
        for line in tqdm(f):
            try:
                if line.startswith("ID"):
                    entry = {"ID": line.split()[1]}
                elif line.startswith("AC"):
                    entry["Accession"] = line.split()[1].strip(";")
                elif line.startswith("DE"):
                    entry.setdefault("Description", []).append(line.strip())
                elif line.startswith("GN"):
                    entry["Gene"] = line.split("=")[1].strip(";")
                elif line.startswith("OS"):
                    entry["Organism"] = line[5:].strip()
                elif line.startswith("SQ"):
                    entry["Sequence"] = []
                elif line.startswith("     ") and "Sequence" in entry:
                    entry["Sequence"].append(line.strip())
                elif line.startswith("//"):
                    if "Sequence" in entry:
                        entry["Sequence"] = "".join(entry["Sequence"])
                    data.append(entry)
                    entry = {}
            except:
                print('line = ', line)
                pass
    return data


def save_original_data_to_csv():
    # Step 1: Download the file
    url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz"
    local_filename = "data/uniprot_sprot.dat.gz"

    # Download the file
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    # Step 2: Decompress the file
    decompressed_file = "data/uniprot_sprot.dat"
    with gzip.open(local_filename, 'rb') as f_in:
        with open(decompressed_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"File decompressed to {decompressed_file}")

    # Parse the decompressed file
    parsed_data = parse_uniprot_flatfile(decompressed_file)
    # Write parsed data to CSV
    csv_file = FILE_PATH
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Accession", "Description", "Gene", "Organism", "Sequence"])
        writer.writeheader()
        writer.writerows(parsed_data)

    print(f"Data saved to {csv_file}")
    return

def read_data():
    if not os.path.exists(FILE_PATH):
        save_original_data_to_csv()
    return pd.read_csv(FILE_PATH)

def preprocess_data(df):
    # Keep only sequence and functional class columns
    df = df[["Sequence", "Function"]]
    df = df.dropna()
    df = df[df["Sequence"].str.len() <= 500]  # Truncate long sequences
    return df

def encode_sequences(df):
    # Tokenize sequences
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(df["Sequence"])
    X = tokenizer.texts_to_sequences(df["Sequence"])
    X = np.array([np.pad(seq, (0, 500 - len(seq)), mode="constant") for seq in X])

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Function"])
    y = to_categorical(y)

    return X, y, tokenizer, label_encoder
