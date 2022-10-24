import gym
import breeding_gym  # noqa: F401
from breeding_gym.wrappers import ObserveStepWrapper
from stable_baselines3 import DQN, PPO
from breeding_gym.utils.paths import DATA_PATH
import wandb
from wandb.integration.sb3 import WandbCallback

from networks.features_extractors import NoFeaturesExtractor
from wrappers.continuous_wrapper import ContinuousWrapper
from wrappers.one_step_ep import OneStepEpWrapper
from wrappers.print_action import PrintActionWrapper


wandb.login(key='eb458e621dd4d01128d5e91ef26c84ddcc82a24e')

run = wandb.init(
    project="sb3",
    settings=wandb.Settings(start_method='fork'),
    config={},
    sync_tensorboard=True,
    save_code=True,
)


train_env = gym.make("KBestBreedingGym",
                     initial_population=DATA_PATH.joinpath("small_geno.txt"),
                     genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
                     individual_per_gen=200,
                     )
eval_env = gym.make("KBestBreedingGym",
                     initial_population=DATA_PATH.joinpath("small_geno.txt"),
                     genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
                     individual_per_gen=200,
                     )

train_env = ObserveStepWrapper(train_env)
train_env = ContinuousWrapper(train_env)
eval_env = ObserveStepWrapper(eval_env)
eval_env = ContinuousWrapper(eval_env)
eval_env = PrintActionWrapper(eval_env)
# train_env = OneStepEpWrapper(train_env)

policy_kwargs = dict(
    features_extractor_class=NoFeaturesExtractor,
    net_arch=[16]
)

#model = DQN('MlpPolicy', train_env, buffer_size=int(1e6), verbose=1, tensorboard_log="runs/test")
model = PPO('MlpPolicy', train_env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="runs/test")
model.learn(total_timesteps=int(1e6), callback=WandbCallback(), log_interval=1, eval_env=eval_env, eval_freq=10_000, n_eval_episodes=100)
model.save("ppo_non_linear")


test_env = gym.make("KBestBreedingGym",
               render_mode="matplotlib",
               initial_population=DATA_PATH.joinpath("small_geno.txt"),
               genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
               individual_per_gen=200,
               render_kwargs={"episode_names": ["Steven's baseline", "RL"]}
               )


obs = test_env.reset()
for i in range(10):
    obs, reward, truncated, terminated, info = test_env.step(10)

obs = test_env.reset()
for i in range(10):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, truncated, terminated, info = test_env.step(action)
    print(action, sep=", ")

test_env.render(file_name="train.png")

run.finish()