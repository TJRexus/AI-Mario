import retro
import gymnasium as gym

env = retro.make(game="SuperMarioBros-Nes")
env.reset()

while True:
    action = env.action_space.sample()
    obs, reward, term, trun, info = env.step(action)
    env.render()
    if term or trun:
        env.reset()
env.close()        