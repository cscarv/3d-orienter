import argparse
from pathlib import Path

def generate_index_recursive(input_dir, output_file):
    input_path = Path(input_dir)
    output_path = Path(output_file)

    if not input_path.is_dir():
        raise ValueError(f"Input path '{input_dir}' is not a valid directory.")

    # Recursively find all .obj files
    obj_files = sorted(input_path.rglob("*.obj"))

    with output_path.open("w") as f:
        for obj_file in obj_files:
            f.write(f"{obj_file.as_posix()}\n")

    print(f"Wrote {len(obj_files)} .obj file paths to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively index .obj files in a directory.")
    parser.add_argument("--input_dir", help="Path to the root directory containing .obj files.")
    parser.add_argument("--output_file", help="Path to the output .txt file.")
    args = parser.parse_args()

    generate_index_recursive(args.input_dir, args.output_file)