import numpy as np
from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import set_random_seed #Allows us to reproduce results so that we tell if changing learning rate changes things
from stable_baselines3.common.callbacks import BaseCallback, CallbackList #Be able to check back in with us
from stable_baselines3.common.vec_env import VecMonitor
import os
import retro
from retro.examples.discretizer import Discretizer
import gymnasium as gymn

class SaveOnBestTrainingRewardCallback(BaseCallback):
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

class NoProgressCallback(BaseCallback):
    def __init__(self, penalty=-0.1, patience=5,verbose=0):
        super().__init__(verbose)
        self.penalty = penalty
        self.patience = patience
        self.no_prog = [0]  # one counter per subenv
        self.last_x = [None]

    def _on_training_start(self):
        # Setup once we know number of environments
        n_envs = self.training_env.num_envs
        self.no_prog = [0 for _ in range(n_envs)]
        self.last_x = [None for _ in range(n_envs)]

    def _on_step(self) -> bool:
        infos = self.locals["infos"]

        for idx, info in enumerate(infos):
            x = info.get("xscrollLo", None)
            if x is not None:
                if self.last_x[idx] is not None and x == self.last_x[idx]:
                    self.no_prog[idx] += 1
                else:
                    self.no_prog[idx] = 0
                self.last_x[idx] = x

                if self.no_prog[idx] >= self.patience:
                    self.locals["rewards"][idx] += self.penalty
        return True

class EntropyDecayCallback(BaseCallback):
    def __init__(self, initial_reward_threshold, increment, entropy_decay_rate, min_entropy_coef, initial_entropy_coef, log_dir, check_freq=2048):
        super(EntropyDecayCallback, self).__init__()
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.reward_threshold = initial_reward_threshold
        self.increment = increment
        self.entropy_decay_rate = entropy_decay_rate
        self.min_entropy_coef = min_entropy_coef
        self.initial_entropy_coef = initial_entropy_coef
        self.best_mean_reward = -float('inf')
        self.step_count = 0
        self.model = None

    def _on_training_start(self) -> None:
        self.model = self.model
        self.model.ent_coef = self.initial_entropy_coef
        return True
    
    def _on_step(self) -> bool:
        try:
            self.step_count += 1
            if(self.step_count % 100 == 0):
                print(self.step_count)

            if(self.step_count >= self.check_freq-1):
                # Retrieve training reward
                x, y = ts2xy(load_results(self.log_dir), "timesteps")
                if len(x) > 0:
                    # Mean training reward over the last 100 episodes
                    mean_reward = np.mean(y[-100:])
                    print(f"The mean reward is {mean_reward:.2f}")
                    if mean_reward > self.reward_threshold:
                        # Increase threshold multiplicatively
                        self.reward_threshold *= self.increment
                        # Decay the entropy coefficient, but don't drop below min_entropy_coef
                        new_ent_coef = max(self.model.ent_coef - self.entropy_decay_rate, self.min_entropy_coef)
                        self.model.ent_coef = new_ent_coef
                        print(f"New Reward Threshold: {self.reward_threshold:.2f}, New Entropy Coefficient: {new_ent_coef:.8f}")
                self.step_count = 0
            
            return True
        except Exception as e:
            print(f"An error of type {type(e).__name__} occurred: {e}")    

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

class MarioDiscretizer(Discretizer):
    def __init__(self,env):
        super().__init__(env=env, combos=[
            [],                             # NOOP
            ['RIGHT'],                      # Move Right
            ['LEFT'],                       # Move Left
            ['A'],                          # Short Jump
            ['RIGHT', 'A'],                 # Run Right + Jump (short)
            ['RIGHT', 'B'],                 # Run Right + Dash
            ['RIGHT', 'B', 'A'],            # Run Right + Dash + Jump (short)
            ['A', 'A', 'A'],                # Long Jump (simulates holding A)
            ['RIGHT', 'B', 'A', 'A', 'A'],  # Run Right + Dash + Jump (long)
        ])

class JumpRewardWrapper(gymn.Wrapper):
    def __init__(self, env, jump_bonus=0.1):
        super().__init__(env)
        self.jump_bonus = jump_bonus
        # Define which discrete actions count as jumps
        self.jump_actions = {3, 7, 8}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # action here is an array (vectorized) or int if single env
        # For vectorized: action = [a] ; for DummyVecEnv it's an int.
        a = int(action[0]) if isinstance(action, (list, tuple, np.ndarray)) else int(action)
        if a in self.jump_actions:
            reward += self.jump_bonus
        return obs, reward, terminated, truncated, info
    
class NoProgressPenaltyWrapper(gymn.Wrapper):
    def __init__(self, env, penalty=-0.1, patience=5):
        super().__init__(env)
        self.penalty = penalty
        self.patience = patience
        self.no_progress_steps = 0
        self.last_xscrollLo = None

    def step(self, action):
        # Step through all existing wrappers
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Unwrap completely to get the raw RetroEnv with .ram
        raw = self.env.unwrapped
        xscrollLo = raw.ram[0x06D5]

        if self.last_xscrollLo is not None and xscrollLo == self.last_xscrollLo:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0

        self.last_xscrollLo = xscrollLo

        if self.no_progress_steps >= self.patience:
            reward += self.penalty

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Reset the counter on new episode
        self.no_progress_steps = 0
        self.last_xscrollLo = None
        return self.env.reset(**kwargs)

class MarioWrapper(gymn.Wrapper):
    def __init__(self, env, seed=0,jump_bonus=0.1,combos=None, skip=2):
        super().__init__(env)
        
        # 1) Seeding
        try:
            env.seed(seed)
        except AttributeError:
            np.random.seed(seed)

        # 2) Jump bonus
        self.jump_bonus = jump_bonus
        self.prev_ctrl = 0
        self.A_MASK = 0x01  # A button

        # 4) Discretizer setup
        if combos is None:
            combos = [
                [],  # NOOP
                ['RIGHT'],
                ['RIGHT', 'A'],      # jump while moving
                ['RIGHT', 'B'],      # run
                ['RIGHT', 'A', 'B'], # run and jump
                ['A'],               # jump in place
                ['LEFT'],
                ['LEFT', 'A']
            ]
        # build action map
        self.buttons = self.env.unwrapped.buttons
        self.action_map = []
        for combo in combos:
            arr = np.array([False]*len(self.buttons))
            for b in combo:
                arr[self.buttons.index(b)] = True
            self.action_map.append(arr)
        self.action_space = gymn.spaces.Discrete(len(self.action_map))

        # 5) Frame skip 
        self.skip = skip

    def reset(self, **kwargs):
        self.no_prog_steps = 0
        self.last_x = None
        self.prev_ctrl = 0
        return self.env.reset(**kwargs)

    def step(self, action_idx):
        # map discrete idx → raw button array
        buttons = self.action_map[action_idx]
        total_reward = 0.0
        done = False
        info = {}

        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(buttons)
            done = terminated or truncated
            total_reward += reward

            # Grab ram to determine no_progress
            ram = info.get("ram")
            if ram is not None:
                xscrollLo = ram[0x06D5]
                info["xscrollLo"] = xscrollLo

            # 1) Jump bonus
            ctrl = info.get('controller_state', None)
            if ctrl is not None:
                if (ctrl & self.A_MASK) and not (self.prev_ctrl & self.A_MASK):
                    total_reward += self.jump_bonus
                self.prev_ctrl = ctrl

            if done:
                break

        return obs, total_reward, terminated, truncated, info

def make_env(env_id, rank, seed=0):
    def _init():
        print(f"[Process {rank}]: Starting environment creation with seed {seed + rank}")
        env = retro.make(game=env_id,state="Level1-1",scenario='./custom_scenario.json')
        env = MarioWrapper(env,seed=seed+rank,jump_bonus=0.1,skip=4)
        print(f"[Process {rank}] Environment creation complete")
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    # Define environment ID and the model path (adjust these as needed)
    env_id = "SuperMarioBros-Nes"
    model_path = "./models/sch-3m.zip"

    # Create a DummyVecEnv for testing (using a single environment)
    env = VecMonitor(DummyVecEnv([make_env(env_id, 0, seed=0)]),"./tmp/monitor/")  # Monitor wrapper for logging

    # Load the saved model and attach the environment
    model = PPO.load(model_path, env=env, device="cpu")

    # Run testing for a few episodes
    num_episodes = 5
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            # Use deterministic policy for evaluation
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]  # In a DummyVecEnv, reward is a one-element array

            env.render()
        print(f"Episode {episode + 1} finished with reward: {total_reward}")

    env.close()
