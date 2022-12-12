config_kbest_transf = dict(
    main="simplified_rl_action",
    name="tune_params",
    env_name="breeding_gym:KBestBreedingGym",
    n_envs=4,
    n_eval_envs=1,
    data_size="small",
    group="KBestAttention",
    project="BreedingGym",
    disable_wandb=False,
    n_launches=1,
    sweep_id="f7yelb36"
)

config_kbest = dict(
    main="simplified_rl_action",
    name="final_launch",
    env_name="breeding_gym:KBestBreedingGym",
    n_envs=4,
    n_eval_envs=1,
    data_size="small",
    group="KBestBreedingGym",
    project="BreedingGym",
    disable_wandb=False,
    profile=False,
    gamma=0.84,
    lr=2e-4,
    lambd=0.94, 
    batch_size=64,
    num_epoch=4,
    buffer_size=262144,
    n_launches=4,
)

config_selection = dict(
    main="selection_index_run",
    name="selection_index",
    data_size="small",
    group="SelectionIndex",
    project="BreedingGym"
)

config_profiling = dict(
    main="profile",
    name="profile",
    data_size="small",
    group="profiling",
    disable_wandb=True
)