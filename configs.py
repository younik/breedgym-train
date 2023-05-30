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
    name="test",
    data_size="small",
    group="SelectionMaskedGEBV",
    project="BreedingGym",
    n_envs=16,
    n_eval_envs=1,
    n_launches=1,
    #sweep_id="cd8jybd9"
    # gamma=0.8122,
    # learning_rate=2e-5,
    # gae_lambda=0.9036,
    # n_epochs=5,
    gamma=0.92,
    learning_rate=5e-4,
    gae_lambda=0.96,
    n_epochs=7,
    total_timesteps=1e6,
)

selection_block_mlp = dict(
    main="selection_index_run",
    name="test",
    data_size="small",
    group="BlockMLP",
    project="BreedingGym",
    n_envs=16,
    n_eval_envs=1,
    n_launches=1,
    total_timesteps=3e5,
    sweep_id="1bn9p5n2"
)

masked_gebv_cnn = dict(
    main="selection_index_run",
    name="tune_params2",
    data_size="small",
    group="CNNTuning",
    project="BreedingGym",
    n_envs=16,
    n_eval_envs=1,
    n_launches=8,
    total_timesteps=3e5,
    sweep_id="1zebdvhb",
)

max_min_mrks = dict(
    main="selection_index_run",
    name="tune_params",
    data_size="small",
    group="MaxMinMarkers",
    project="BreedingGym",
    n_envs=16,
    n_eval_envs=1,
    n_launches=4,
    total_timesteps=1e6,
    gae_lambda=0.9090136974853602,
    gamma=0.9667289143098434,
    learning_rate=0.00010803780886009062,
    n_epochs=5
    #sweep_id="toxeze1h",
)

cross_as_paper = dict(
    main="selection_index_run",
    name="tune_params",
    data_size="small",
    group="CrossAsPaper",
    project="BreedingGym",
    n_envs=16,
    n_eval_envs=1,
    n_launches=8,
    total_timesteps=3e5,
    sweep_id="k10uthhd",
)

pair_score = dict(
    main="pair_score_run",
    name="tune_params",
    data_size="small",
    group="PairScore",
    project="BreedingGym",
    n_envs=4,
    n_eval_envs=1,
    n_launches=1,
    total_timesteps=1e6,
    sweep_id="yb6cz5x1"
)


minimal_pair_score = dict(
    main="pair_score_run",
    name="minal_pair_score",
    genome="small_const_chr_geno.npy",
    genetic_map="small_const_chr_genetic_map.txt",
    group="HistogramizePairScore",
    project="BreedingGym",
    n_envs=8,
    n_eval_envs=4,
    n_launches=8,
    total_timesteps=10e6,
    timeout=120,
    # gae_lambda=0.9090136974853602,
    # gamma=0.9667289143098434,
    # learning_rate=0.00010803780886009062,
    # n_epochs=5,
    kernel_size1=64,
    stride1=8,
    kernel_size2=4,
    stride2=1,
    out_channels2=8,
    gen_features_dim=8,
    sweep_id="y7exi3ky"
)

evaluate_saved_model = dict(
    main="evaluate_model",
    name="evaluate",
    genome="sample_full_pop_geno.npy",
    genetic_map="sample_with_r_genetic_map.txt",
    group="TrainICML",
    project="BreedingGym",
    n_eval_envs=8,
    gpus=0,
    debug=True
)

supervised_training = dict(
    main="supervised_cnn_train",
    name="supervised_10k",
    genetic_map="small_const_chr_genetic_map.txt",
    group="SupervisedTraining",
    project="BreedingGym",
    kernel_size1=64,
    stride1=8,
    kernel_size2=4,
    stride2=1,
    out_channels2=8,
    features_dim=256
)

baseline_run = dict(
    main="baseline",
    name="baseline",
    group="Baseline",
    project="BreedingGym",
    genome="sample_full_pop_geno.npy",
    genetic_map="sample_with_r_genetic_map.txt",
    n_envs=16,
    debug=True,   
)


factored_ppo = dict(
    main="factored_ppo_run",
    name="mean_remove_rew_bounds",
    genome="small_const_chr_geno.npy",
    genetic_map="small_const_chr_genetic_map.txt",
    group="FactoredPPO",
    project="BreedingGym",
    num_generations=10,
    n_envs=16,
    n_eval_envs=4,
    n_launches=8,
    total_timesteps=2e6,
    timeout=120,

    # gae_lambda=0.9102172650637604,
    # gamma=0.9013120188542332,
    # learning_rate=0.00003282341903073194,
    # n_epochs=4,
    # ent_coeff=0,

    num_bins=512,
    kernel_size1=64,
    stride1=8,
    kernel_size2=4,
    stride2=1,
    out_channels2=8,
    gen_features_dim=8,
    sweep_id="0kv6jvza",
)


wheat_data = dict(
    main="selection_index_run",
    name="all_together",
    genome="wheat_genome.npy",
    genetic_map="wheat_genetic_map.csv",
    group="WheatData",
    project="BreedingGym",
    num_generations=10,
    n_envs=8,
    n_eval_envs=4,
    n_launches=8,
    total_timesteps=5e6,
    timeout=120,

    # gae_lambda=0.935,
    # gamma=0.93,
    # learning_rate=1.5e-5,
    # n_epochs=6,
    # ent_coeff=0,

    num_bins=128,
    kernel_size1=16,
    stride1=16,
    kernel_size2=4,
    stride2=4,
    out_channels2=8,
    gen_features_dim=4,
    sweep_id="dqovddqo",
)

minimal_1k = dict(
    main="selection_index_run",
    name="train_max_no_curriculum",
    genome="sample_full_pop_geno.npy",
    genetic_map="sample_with_r_genetic_map.txt",
    group="TrainICMLMax",
    project="BreedingGym",
    num_generations=10,
    n_envs=8,
    n_eval_envs=4,
    n_launches=1,
    total_timesteps=10e6,
    timeout=120,

    gae_lambda=0.9332137965259578,
    gamma=0.9665589863965516,
    learning_rate=0.00001211489234708656,
    n_epochs=4,
    ent_coeff=0.00032598421377001116,
    kernel_size1=32,
    stride1=8,
    out_channels1=64,
    kernel_size2=8,
    stride2=2,
    out_channels2=16,
    gen_features_dim=16,
    policy_hiddens=32,
    value_hiddens=32,
)


wheat_bp = dict(
    main="wheat_run",
    name="test_wheat2",
    genome="sample_full_pop_geno.npy",
    genetic_map="sample_with_r_genetic_map.txt",
    group="WheatBP",
    project="BreedingGym",
    num_generations=10,
    n_envs=4,
    n_eval_envs=4,
    n_launches=16,
    total_timesteps=10e6,
    timeout=120,

    gae_lambda=0.9332137965259578,
    gamma=0.9665589863965516,
    learning_rate=0.00001211489234708656,
    n_epochs=4,
    ent_coeff=0.00032598421377001116,
    kernel_size1=32,
    stride1=8,
    out_channels1=64,
    kernel_size2=8,
    stride2=2,
    out_channels2=16,
    gen_features_dim=16,
    policy_hiddens=32,
    value_hiddens=32,
    sweep_id="4jiu1iqi"
)