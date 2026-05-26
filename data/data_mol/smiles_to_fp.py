import argparse
import os
import sys
import pandas as pd
import numpy as np
try:
    from rdkit import Chem
except ImportError:
    print("Error: RDKit is not installed. Please install it using 'pip install rdkit' or run this in your scDrug docker container.")
    sys.exit(1)
def smiles_to_fp(smiles_list):
    """
    Convert a list of SMILES strings to 2048-bit RDKit fingerprints.
    """
    fps = []
    valid_indices = []
    
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Invalid SMILES ignored: {smiles}")
            continue
        
        # Generate the standard RDKit fingerprint and convert to a list of ints (0 or 1)
        bit_string = Chem.RDKFingerprint(mol).ToBitString()
        fp = [int(bit) for bit in bit_string]
        fps.append(fp)
        valid_indices.append(i)
        
    return np.array(fps), valid_indices
def main():
    parser = argparse.ArgumentParser(description="Convert SMILES strings to 2048-bit RDKit Fingerprints CSV.")
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-s', '--smiles', nargs='+', help='List of SMILES strings directly from the command line.')
    group.add_argument('-i', '--input', help='Path to an input CSV file containing SMILES.')
    
    # Configuration options
    parser.add_argument('-o', '--output', default='fingerprints.csv', help='Path to the output CSV file (default: fingerprints.csv).')
    parser.add_argument('--smiles-col', default='smiles', help='Name of the SMILES column if reading from CSV (default: smiles).')
    parser.add_argument('--name-col', default='mol_name', help='Name of the molecule identifier column if reading from CSV (default: mol_name).')
    
    args = parser.parse_args()
    
    smiles_to_convert = []
    identifiers = []
    
    # Load SMILES input
    if args.smiles:
        smiles_to_convert = args.smiles
        identifiers = [f"mol_{i+1}" for i in range(len(smiles_to_convert))]
    else:
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} does not exist.")
            sys.exit(1)
            
        try:
            df = pd.read_csv(args.input)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            sys.exit(1)
            
        if args.smiles_col not in df.columns:
            print(f"Error: Column '{args.smiles_col}' not found in CSV. Available columns: {list(df.columns)}")
            sys.exit(1)
            
        smiles_to_convert = df[args.smiles_col].astype(str).tolist()
        if args.name_col in df.columns:
            identifiers = df[args.name_col].astype(str).tolist()
        else:
            identifiers = [f"mol_{i+1}" for i in range(len(smiles_to_convert))]
            
    if not smiles_to_convert:
        print("No SMILES to process.")
        sys.exit(1)
        
    print(f"Converting {len(smiles_to_convert)} SMILES to RDKit fingerprints...")
    fps, valid_indices = smiles_to_fp(smiles_to_convert)
    
    if len(fps) == 0:
        print("Error: No valid fingerprints could be generated.")
        sys.exit(1)
        
    # Filter identifiers for only valid SMILES
    valid_identifiers = [identifiers[i] for i in valid_indices]
    valid_smiles = [smiles_to_convert[i] for i in valid_indices]
    
    # Create the output dataframe
    df_out = pd.DataFrame(fps, columns=[f"bit_{i}" for i in range(fps.shape[1])])
    df_out.insert(0, 'smiles', valid_smiles)
    df_out.insert(0, 'mol_name', valid_identifiers)
    
    # Save to CSV
    try:
        df_out.to_csv(args.output, index=False)
        print(f"Fingerprints successfully saved to {args.output}")
        print(f"Shape: {df_out.shape} (Rows: molecules, Columns: identifier + SMILES + 2048 fingerprint bits)")
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)
if __name__ == '__main__':
    main()