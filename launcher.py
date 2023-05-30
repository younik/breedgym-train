import subprocess
import tempfile
import os
import pathlib
import logging
import configs

config = configs.minimal_1k


def get_launch_args():
    args = config.items()
    args = filter(lambda k_v: not isinstance(
        k_v[1], bool) or k_v[1], args)  # delete --key=False
    args = filter(lambda k_v: k_v[1] is not None, args)
    return ' '.join(f'--{key}={value}' for key, value in args)


def template(name, stdout_path, nodes=1, cpus=1, gpus=1, memory=100_000, timeout=120, **kwargs):
    return f"""#!/bin/bash

#SBATCH --job-name={name}
#SBATCH --output={stdout_path}/{name}_%j.out
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem-per-cpu={memory}
{f"#SBATCH --gpus-per-node={gpus}" if gpus > 0 else ""}
#SBATCH --time={timeout}:00:00

srun python main.py {get_launch_args()} --jobid=$SLURM_JOBID
"""


def make_paths():
    path = pathlib.Path(__file__).parent.joinpath("outs", config["group"])
    path.mkdir(parents=True, exist_ok=True)
    config["path"] = path

    stdout_path = path.joinpath("stdout")
    stdout_path.mkdir(exist_ok=True)
    config["stdout_path"] = stdout_path

    fig_path = path.joinpath("figures")
    fig_path.mkdir(exist_ok=True)
    config["fig_path"] = fig_path

    model_path = path.joinpath("models")
    model_path.mkdir(exist_ok=True)
    config["model_path"] = model_path

    log_path = path.joinpath("logs")
    log_path.mkdir(exist_ok=True)
    config["log_path"] = log_path

    profile_path = path.joinpath("profile")
    profile_path.mkdir(exist_ok=True)
    config["profile_path"] = profile_path


def map_data_size():
    data_map = {
        "sample": dict(
            genome="sample_geno.npy",
            genetic_map="sample_genetic_map.txt",
        ),

        "sample_full_pop": dict(
            genome="sample_full_pop_gen.npy",
            genetic_map="sample_genetic_map.txt",
        ),

        "sample_with_r": dict(
            genome="sample_geno.npy",
            genetic_map="sample_with_r_genetic_map.txt",
        ),

        "small": dict(
            genome="small_geno.npy",
            genetic_map="small_genetic_map.txt",
        ),

        "medium": dict(
            genome="medium_geno.npy",
            genetic_map="medium_genetic_map.txt",
        ),

        "big": dict(
            genome="big_geno.npy",
            genetic_map="big_genetic_map.txt",
        ),

        "full": dict(
            genome="geno.npy",
            genetic_map="genetic_map.txt",
        ),
    }

    if "data_size" in config.keys():
        config.update(data_map[config["data_size"]])


def sbatch_launch(n_launches):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".sbatch", delete=False)
    f.write(template(**config))
    f.close()
    for _ in range(n_launches):
        subprocess.call(["sbatch", f.name])
    os.unlink(f.name)


def srun_launch(nodes=1, cpus=1, gpus=1, memory=100_000, **kwargs):
    args = [
        f"--nodes={nodes}",
        f"--cpus-per-task={cpus}",
        f"--mem-per-cpu={memory}"
    ]
    if gpus > 0:
        args.append(f"--gpus-per-node={gpus}")

    args.append(f"python main.py {get_launch_args()}")
    subprocess.call(' '.join(["srun", *args]), shell=True)


if __name__ == "__main__":
    make_paths()
    map_data_size()
    n_launches = config.pop("n_launches", 1)

    debug_mode = config.get("debug", False)
    if debug_mode and not n_launches != 1:
        logging.warning("Debug mode: ignoring n_launches argument.")
    if debug_mode and not config.get("disable_wandb", False):
        logging.warning("Debug mode: disabling Weights and Biases.")
        config["disable_wandb"] = True

    if debug_mode:
        srun_launch(**config)
    else:
        sbatch_launch(n_launches)
