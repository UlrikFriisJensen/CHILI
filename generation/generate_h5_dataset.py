from h5_constructor import h5Constructor
import argparse

def main(args):
    
    # Read paths from batch
    with open(args.batch, 'r') as f:
        file_paths = [line.strip() for line in f.readlines()]

    # Generate h5s
    gc = h5Constructor(save_dir = args.output, cif_paths=file_paths)
    gc.gen_h5s(parallelize=False, device='cuda')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', '-b', required=True, type=str)
    parser.add_argument('--output', '-o', required=True, type=str)
    args = parser.parse_args()
    main(args)

