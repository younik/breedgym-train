from stable_baselines3.common.callbacks  import EvalCallback


class MyEvalCallback(EvalCallback):

    def _log_success_callback(self, locals_, globals_) -> None:
        print("called log", flush=True)
        
        info = locals_["info"]
        
        self.logger.record("eval/GEBVs", info["GEBV"])