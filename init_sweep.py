import wandb

project_name = "BreedingGym"

sweep_config = {
    'name': 'attention',
    'method': 'bayes',
    'metric': {
        'goal': 'maximize', 
        'name': 'eval/mean_reward'
        },
    'parameters': {
        'gamma': {'min': 0.8, 'max': 0.995},
        'gae_lambda': {'min': 0.9, 'max': 0.95},
        'buffer_size': {"values": [2**x for x in range(11, 19)]},
        'batch_size': {"values": [2**x for x in range(5, 10)]},
        'n_epochs': {'min': 3, 'max': 10},
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3
        }
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)