import numpy as np
import time
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor


class FactoredVecMonitor(VecMonitor):
    
    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_returns += np.mean(rewards, axis=1)
        self.episode_lengths += 1
        new_infos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {"r": episode_return, "l": episode_length, "t": round(time.time() - self.t_start, 6)}
                for key in self.info_keywords:
                    episode_info[key] = info[key]
                info["episode"] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(episode_info)
                new_infos[i] = info
        return obs, rewards, dones, new_infos