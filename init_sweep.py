import wandb

project_name = "BreedingGym"

# sweep_config = {
#     'name': 'pair_score_bn',
#     'method': 'bayes',
#     'metric': {
#         'goal': 'maximize', 
#         'name': 'eval/mean_reward'
#         },
#     'parameters': {
#         'gamma': {'min': 0.85, 'max': 0.995},
#         'gae_lambda': {'min': 0.9, 'max': 0.95},
#         #'buffer_size': {"values": [2**x for x in range(11, 19)]},
#         #'batch_size': {"values": [2**x for x in range(5, 10)]},
#         "value_hiddens": {"values": [[512], [128, 32], [512, 64], [512, 256, 64]]},
#         'n_epochs': {'min': 3, 'max': 10},
#         'learning_rate': {
#             'distribution': 'log_uniform_values',
#             'min': 1e-6,
#             'max': 1e-4
#         }
#     }
# }


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

sweep_config = {
    'name': 'pair_score_a2c',
    'method': 'bayes',
    'metric': {
        'goal': 'maximize', 
        'name': 'eval/mean_reward'
        },
    'parameters': {
        'gamma': {'min': 0.9, 'max': 1.0},
        'gae_lambda': {'min': 0.95, 'max': 1.0},
        "value_hiddens": {"values": [[128, 32], [512, 64], [512, 256, 64]]},
        'ent_coef': {"values": [0, 0.05, 0.1, 0.25]},
        'n_steps': {"values": [10, 20, 30]},
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-3
        }
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)