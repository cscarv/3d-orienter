import os
import argparse
import subprocess
import json5

if __name__ == "__main__":
    # add project base directory to path. adjust number of .parent's to fit file structure
    import sys
    from pathlib import Path
    proj_dir = str(Path(__file__).resolve().parent.parent)
    sys.path.append(proj_dir)

    # parse specs
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", "-e", default="config/default", help="Path to specs.json5")
    args = parser.parse_args()    
    actual_exp_dir = args.exp_dir.rstrip(" /")
    
    # read node/gpu count from specs
    with open(os.path.join(actual_exp_dir, "specs.json5"), "r") as file:
        specs = json5.load(file)
    num_nodes = specs["num_compute_nodes"]
    num_gpus_per_node = specs["num_gpus_per_compute_node"]
    num_cpus_per_gpu = specs["cpus_per_gpu"]
    
    # slurm file output path
    output_path = os.path.join(proj_dir, "slurm_output_files", "%j_%x.out")

    # build sbatch command
    job_name = "orienter-3d"
    command = f"sbatch -n {num_nodes} --job-name={job_name} --output={output_path} --cpus-per-gpu={num_cpus_per_gpu} --ntasks-per-node={num_gpus_per_node} --gpus-per-node={num_gpus_per_node} launch_internals/slurm_script.sh {actual_exp_dir}"
    print(f"COMMAND: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        print(f"Successfully submitted sbatch job:\n{stdout.decode('utf-8')}")
    else:
        print(f"Error submitting sbatch job:\n{stderr.decode('utf-8')}")