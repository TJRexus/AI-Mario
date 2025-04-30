import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
#from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import set_random_seed #Allows us to reproduce results so that we tell if changing learning rate changes things
from stable_baselines3.common.callbacks import BaseCallback, CallbackList #Be able to check back in with us
from stable_baselines3.common.vec_env import VecMonitor
from torch.utils.tensorboard import SummaryWriter
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

class CustomEntropySchedule:
    def __init__(self, init_entropy=0.02, min_entropy=0.001, decay_rate=0.95, reward_threshold=100, increment=1.2):
        self.current_entropy = init_entropy
        self.min_entropy = min_entropy
        self.decay_rate = decay_rate
        self.reward_threshold = reward_threshold
        self.increment = increment

    def __call__(self) -> float:
        return self.current_entropy

    def update(self, mean_reward):
        if mean_reward > self.reward_threshold:
            self.reward_threshold *= self.increment
            self.current_entropy = max(
                self.current_entropy * self.decay_rate,
                self.min_entropy
            )
            print(f"Entropy Decayed: new_entroy={self.current_entropy:.6f}, new_threshold={self.reward_threshold:.2f}")
        return self.current_entropy    

class CustomLearningRateSchedule:
    def __init__(self, initial_lr=2.5e-4, min_lr=1e-5, decay_rate=0.95, reward_threshold=100, increment=1.2):
        self.current_lr = initial_lr
        self.min_lr = min_lr
        self.decay_rate = decay_rate
        self.reward_threshold = reward_threshold
        self.increment = increment

    def __call__(self) -> float:
        return self.current_lr

    def update(self, mean_reward):
        if mean_reward > self.reward_threshold:
            self.reward_threshold *= self.increment
            self.current_lr = max(
                self.current_lr * self.decay_rate,
                self.min_lr
            )
            print(f"Learning Rate decayed: new_lr={self.current_lr:.8f}, next_threshold={self.reward_threshold:.2f}")
        return self.current_lr    

class ScheduleDecayCallback(BaseCallback):
    def __init__(self, entropy_schedule, lr_schedule, log_dir, check_freq=1024):
        super().__init__()
        self.entropy_schedule = entropy_schedule
        self.lr_schedule = lr_schedule
        self.log_dir = log_dir
        self.check_freq = check_freq
        self.step_count = 0
        self.writer = None
    
    def _on_training_start(self):
        self.writer = SummaryWriter(log_dir=os.path.join("./board", "entropy_logs"))
        return True

    def _on_step(self) -> bool:
        try:
            self.step_count += 1
            if(self.step_count % 100 == 0):
                print(self.step_count)

            if(self.step_count >= self.check_freq):
                # Retrieve training reward
                x, y = ts2xy(load_results(self.log_dir), "timesteps")
                if len(x) > 0:
                    # Mean training reward over the last 100 episodes
                    mean_reward = np.mean(y[-100:])
                    print(f"The mean reward is {mean_reward:.2f}")
                    self.model.ent_coef = self.entropy_schedule.update(mean_reward)
                    self.model.learning_rate = self.lr_schedule.update(mean_reward)

                    new_coef = self.entropy_schedule()
                    self.writer.add_scalar("Entropy Coefficient", new_coef, self.num_timesteps)
                    self.writer.add_scalar("Mean Reward", mean_reward, self.num_timesteps)

                self.step_count = 0
            return True
        except Exception as e:
            print(f"An error of type {type(e).__name__} occurred: {e}")    
            self.step_count = 0
        return True
    
    def on_training_end(self):
        if self.writer:
            self.writer.close()
        return True    
        
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
                ['LEFT', 'A'],
                ['DOWN']
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

        # 5) Frame skip (we'll call underlying step multiple times)
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
    log_dir = "tmp/monitor/"
    os.makedirs(log_dir,exist_ok=True) #Checks if our log file exists

    env_id = "SuperMarioBros-Nes"
    num_cpu = 4

    env = VecMonitor(SubprocVecEnv([make_env(env_id,i) for i in range(num_cpu)]), "tmp/monitor/")
    #env = VecMonitor(DummyVecEnv([make_env(env_id, 0)]), "tmp/monitor/")
    venv = VecFrameStack(env, n_stack=4)

    entropy_schedule = CustomEntropySchedule(
        init_entropy=0.02,
        min_entropy=0.001,
        decay_rate = 0.95,
        reward_threshold=300,
        increment=1.2
    )
    lr_schedule = CustomLearningRateSchedule(
        initial_lr=2.5e-4,
        min_lr=1e-5,
        decay_rate=0.95,
        reward_threshold=300,
        increment=1.2
    )
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./board/", learning_rate=lr_schedule.current_lr, n_steps=2048, batch_size=256, ent_coef=entropy_schedule.current_entropy)
    #Use a previous model to not start from scratch
    #model = PPO.load("./tmp/monitor/best_model.zip", env = env)

    print("----------------------------Starting Learning----------------------------")
    best_model_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir) #Checks every 1000 steps to see if it has a better model
    no_progress_callback = NoProgressCallback(penalty=-0.1, patience=5)
    decay_callback = ScheduleDecayCallback(
        entropy_schedule=entropy_schedule,
        lr_schedule=lr_schedule,
        log_dir=log_dir,
        check_freq=1024
    )
    callbacks = CallbackList([best_model_callback,no_progress_callback,decay_callback])
    model.learn(total_timesteps=100000, callback=callbacks, tb_log_name="Prezzy-new-") #Actually starts learning and create tensorboard
    model.save("./models/Prezzy-new-1")
    env.close()
    print("----------------------------Done Learning----------------------------")