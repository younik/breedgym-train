import wandb

project_name = "BreedingGym"

sweep_config = {
    'name': 'wheat_bp2',
    'method': 'bayes',
    'metric': {
        'goal': 'maximize', 
        'name': 'eval/mean_reward'
        },
    'parameters': {
        'gamma': {'min': 0.8, 'max': 0.995},
        'gae_lambda': {'min': 0.8, 'max': 0.95},
        'n_epochs': {'min': 3, 'max': 10},
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 5e-7,
            'max': 1e-4
        },
        'ent_coeff': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-3
        },
        # 'kernel_size1':  {"values": [32, 64, 128]},
        # 'stride1': {"values": [8, 16, 32]},
        # 'out_channels1': {"values": [16, 32, 64]},
        # 'kernel_size2': {"values": [4, 8, 16]},
        # 'stride2': {"values": [1, 2, 4]},
        # 'out_channels2': {"values": [4, 8, 16]},
        # 'gen_features_dim': {"values": [4, 8, 16]},
        # 'policy_hiddens': {"values": [16, 32, 64]},
        # 'value_hiddens': {"values": [16, 32, 64]},
    }
}


# sweep_arch_search = {
#     'name': 'cnn_arch_search',
#     'method': 'random',
#     'metric': {
#         'goal': 'maximize', 
#         'name': 'eval/mean_reward'
#         },
#     'parameters': {
#         "normalize_markers": {"values": [True, False]},
#         "out_channels1": {"values": [16, 64, 128, 256]},
#         "out_channels2": {"values": [1, 4, 16, 32]},
#         "kernel_size1": {"values": [128, 256, 512]},
#         "kernel_size2": {"values": [8, 16, 32, 64]},
#         "stride1": {"values": [32, 64, 128]},
#         "stride2": {"values": [2, 4, 8]},
#         "max_pooling_size": {"values": [1, 4, 8]},
#         "policy_hiddens": {"values": [None, 16, 64]},
#         "value_hiddens": {"values": [None, 16, 64]},
#     }
# }

# sweep_config = {
#     'name': 'pair_score_a2c',
#     'method': 'bayes',
#     'metric': {
#         'goal': 'maximize', 
#         'name': 'eval/mean_reward'
#         },
#     'parameters': {
#         'gamma': {'min': 0.9, 'max': 1.0},
#         'gae_lambda': {'min': 0.95, 'max': 1.0},
#         "value_hiddens": {"values": [[128, 32], [512, 64], [512, 256, 64]]},
#         'ent_coef': {"values": [0, 0.05, 0.1, 0.25]},
#         'n_steps': {"values": [10, 20, 30]},
#         'learning_rate': {
#             'distribution': 'log_uniform_values',
#             'min': 1e-6,
#             'max': 1e-3
#         }
#     }
# }

sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)