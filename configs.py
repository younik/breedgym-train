kbest_transf = dict(
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

kbest = dict(
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

selection_cnn = dict(
    main="selection_index_run",
    name="selection_final",
    data_size="small",
    group="SelectionIndex",
    project="BreedingGym",
    n_envs=32,
    n_eval_envs=1,
    #sweep_id="cjel19ew",
    n_launches=4,
    gamma=0.92,
    learning_rate=5e-4,
    gae_lambda=0.96,
    n_epochs=7
)

selection_append_gebv = dict(
    main="selection_index_run",
    name="final_launch",
    data_size="small",
    group="SelectionAppendGEBV",
    project="BreedingGym",
    n_envs=32,
    n_eval_envs=1,
    n_launches=4,
    #sweep_id="8aswjmox"
    gamma=0.9434,
    learning_rate=5.5e-5,
    gae_lambda=0.9061,
    n_epochs=8,
    total_timesteps=1e6
)

selection_masked_gebv = dict(
    main="selection_index_run",
    name="final_launch_1e6",
    data_size="small",
    group="SelectionMaskedGEBV",
    project="BreedingGym",
    n_envs=32,
    n_eval_envs=1,
    n_launches=1,
    #sweep_id="cd8jybd9"
    gamma=0.8122,
    learning_rate=2e-5,
    gae_lambda=0.9036,
    n_epochs=5,
    total_timesteps=1e6
)

selection_block_mlp = dict(
    main="selection_index_run",
    name="tune_params",
    data_size="small",
    group="BlockMLP",
    project="BreedingGym",
    n_envs=16,
    n_eval_envs=1,
    n_launches=8,
    total_timesteps=3e5,
    sweep_id="1bn9p5n2"
)

masked_gebv_cnn = dict(
    main="selection_index_run",
    name="tune_params",
    data_size="small",
    group="MaskedGEBV_CNN",
    project="BreedingGym",
    n_envs=16,
    n_eval_envs=1,
    n_launches=8,
    total_timesteps=3e5,
    sweep_id="n5l0e7dl"
)