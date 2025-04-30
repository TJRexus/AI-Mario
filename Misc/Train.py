import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import set_random_seed #Allows us to reproduce results so that we tell if changing learning rate changes things
from stable_baselines3.common.callbacks import BaseCallback #Be able to check back in with us
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv #Skips some frames to help it learn better
import os
import retro
import gymnasium as gymn

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          print(f"[Callback] n_calls: {self.n_calls}, num_timesteps: {self.num_timesteps}")
          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

class RetroEnvWithSeed(gymn.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._seed = None

    def reset(self, seed=None, **kwargs):
        print(f"[PID {os.getpid()}]: Reset called with seed {seed}")
        if seed is not None:
            self.seed(seed)
        print(f"[PID {os.getpid()}]: Environment Reset Complete")    
        # Gymnasium’s reset returns (obs, info)
        return self.env.reset(seed=self._seed, **kwargs)

    def step(self, action):
        # Gymnasium's step returns (obs, reward, terminated, truncated, info)
        #print(f"[PID {os.getpid()}] Step called with action: {action}")
        result = self.env.step(action)
        #print(f"[PID {os.getpid()}] Step complete")
        return result
    
    def seed(self, seed=None):
        self._seed = seed
        try:
            s = self.env.seed(seed)
            print(f"[PID {os.getpid()}] Seed set to: {seed}")
            return s
        except AttributeError:
            np.random.seed(seed)
            print(f"[PID {os.getpid()}] Numpy seed set to: {seed}")
            return [seed]      

def make_env(env_id, rank, seed=0):
    def _init():
        print(f"[Process {rank}]: Starting environment creation with seed {seed + rank}")
        env = retro.make(game=env_id,state="Level1-1",scenario='./custom_scenario.json')
        env = RetroEnvWithSeed(env)
        env.seed(seed + rank) #Gives a random environment
        env = MaxAndSkipEnv(env, 4) #Makes it a quicker training process by letting them make a decision every 4 frames
        print(f"[Process {rank}] Environment creation complete")
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    log_dir = "tmp/monitor/"
    os.makedirs(log_dir,exist_ok=True) #Checks if our log file exists

    env_id = "SuperMarioBros-Nes"
    num_cpu = 4

    env = VecMonitor(SubprocVecEnv([make_env(env_id,i) for i in range(num_cpu)]), "tmp/monitor/")
    #env = VecMonitor(DummyVecEnv([make_env(env_id, 0)]), "tmp/monitor/")

    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./board/", learning_rate=0.00003, device='cpu')
    #Use a previous model to not start from scratch
    #model = PPO.load("Path to model", env = env)

    print("----------------------------Starting Learning----------------------------")
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir) #Checks every 1000 steps to see if it has a better model
    model.learn(total_timesteps=100000, callback=callback, tb_log_name="PPO-00003-100000-Death20 ") #Actually starts learning and create tensorboard
    model.save(env_id)
    print("----------------------------Done Learning----------------------------")