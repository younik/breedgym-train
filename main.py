import sys
import traceback
import wandb
import importlib
import argparse
import os
import multiprocessing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--main', type=str, help='File name to launch (without .py)')
    parser.add_argument('--name', type=str, help='Name of the launch')
    parser.add_argument('--group', type=str, help='Group name of the launch')
    parser.add_argument('--project', type=str, help='Project name')
    parser.add_argument('--jobid', type=int, help='Slurm job id')
    parser.add_argument('--env_name', type=str, help='Name of gym environment')
    parser.add_argument('--data_size', type=str, help='Size of data to use')
    parser.add_argument('--path', type=str, help='Path used for this run')
    parser.add_argument('--stdout_path', type=str, help='Path for stdout file')
    parser.add_argument('--fig_path', type=str, help='Path for saving figures')
    parser.add_argument('--model_path', type=str, help='Path for saving models')
    parser.add_argument('--log_path', type=str, help='Path for saving logs')
    parser.add_argument('--profile_path', type=str, help='Path for saving profile traces')
    parser.add_argument('--genome', type=str, help='Genome file')
    parser.add_argument('--genetic_map', type=str, help='Genetic map file')
    parser.add_argument('--disable_wandb', type=bool, default=False, help='Disable WandB')
    parser.add_argument('--sweep_id', type=str, default=None, help='Wandb sweep id of the launch')
    parser.add_argument('--n_envs', type=int, default=1, help='Number of train env')
    parser.add_argument('--n_eval_envs', type=int, default=1, help='Number of eval env')
    parser.add_argument('--profile', type=bool, default=False, help='Profile code')

    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg.split('=')[0], type=float)
            
    config = parser.parse_args()
    config.unique_name = f"{config.name}_{config.jobid}"
    
    def launch():    
        run = None
        if not config.disable_wandb:
            wandb.login()

            run = wandb.init(
                project=config.project,
                name=config.name,
                group=config.group,
                config=vars(config),
                settings=wandb.Settings(start_method='fork'),
                sync_tensorboard=True,
                save_code=True,
            )
            
            vars(config).update(wandb.config)
        
        for key, value in vars(config).items():
            print(f"{key}: {value}")
        print(flush=True)

        # # see https://github.com/google/jax/issues/5506
        # num_cpus = multiprocessing.cpu_count()
        # flag = f"--xla_force_host_platform_device_count{num_cpus}"
        # os.environ["XLA_FLAGS"] = flag
        # see https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        
        import jax
        if config.profile:
            jax.profiler.start_trace(
                f"{config.profile_path}/{config.unique_name}",
                create_perfetto_trace=True
            )
        
        script = importlib.import_module(f"scripts.{config.main}")
        main = getattr(script, "main")
        
        try:
            main(config)
        except Exception as e:
            print(e)
            print(traceback.print_exc(), file=sys.stderr)
        finally:
            if config.profile:
                jax.profiler.stop_trace()
            if run is not None:
                run.finish()

    if not config.disable_wandb and config.sweep_id is not None:
        wandb.agent(config.sweep_id, launch, "younis", config.project, 1)
    else:
        launch()