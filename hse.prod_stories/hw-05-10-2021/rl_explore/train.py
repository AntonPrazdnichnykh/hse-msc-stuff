import os
import sys
import shutil
import torch
import numpy as np
from gym import spaces

from PIL import Image
import wandb
from omegaconf import OmegaConf
from my_ppo import PPO

sys.path.append(os.path.abspath("mapgen"))
os.environ["PYTHONPATH"] = os.path.abspath("mapgen")
from mapgen import Dungeon


class ModifiedDungeon(Dungeon):
    """Use this class to change the behavior of the original env (e.g. remove the trajectory from observation, like here)"""
    def __init__(self,
            width=20,
            height=20,
            max_rooms=3,
            min_room_xy=5,
            max_room_xy=12,
            observation_size=11,
            vision_radius=5,
            max_steps: int = 2000,
            reward: str = 'basic'
    ):
        super().__init__(
            width=width,
            height=height,
            max_rooms=max_rooms,
            min_room_xy=min_room_xy,
            max_room_xy=max_room_xy,
            observation_size = observation_size,
            vision_radius = vision_radius,
            max_steps = max_steps
        )

        self.observation_space = spaces.Box(0, 1, [observation_size, observation_size, 3]) # because we remove trajectory and leave only cell types (UNK, FREE, OCCUPIED)
        self.action_space = spaces.Discrete(3)
        self.reward = reward

    def step(self, action):
        observation, reward , done, info = super().step(action)
        if self.reward == 'complex':
            if info['moved']:
                if info['new_explored'] > 0:  # explore new cell
                    reward = 0.1 + info['total_explored'] / info['total_cells'] * 20
                else:  # just free move
                    reward = -0.5
            else:  # bump into tile
                reward = -1

        # observation = observation[:, :, :-1] # remove trajectory
        return observation, reward , done, info


def fix_seed(env, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)



def train():
    config = {
        'env': {
            'width': 20,
            'height': 20,
            'max_rooms': 3,
            'min_room_xy': 5,
            'max_room_xy': 10,
            'observation_size': 11,
            'vision_radius': 5,
            'reward': 'basic',
        },
        'model': {
            'in_shape': (11, 11),
            'in_channels': 4,
            'out_dim': 3,
            'conv_filters': [
                [16, (3, 3), 2],
                [32, (3, 3), 2],
                [32, (3, 3), 1],
            ],
            'post_fcn_hiddens': [32],
        },
        'training_params': {
            'ppo': {
                'n_epochs': 80,
                'eps_clip': 0.2,
                'gamma': 0.95,
                'lr_actor': 1e-4,
                'lr_critic': 3e-4,
                'entropy_coeff': 0.1,
                'vf_loss_coeff': 1.0
            },
            'other': {
                'max_ep_len': 100,
                'max_training_timesteps': int(3e5),
                'print_freq': 1000,
                'log_freq': 200,
                'render_freq': 10000,
                'save_model_freq': 10000,
                'update_timesteps': 400,
                'random_seed': 0,
            }
        }
    }

    wandb.init(project='prod-stories-05-10', config=config)
    config = OmegaConf.create(config)

    ckpt_dir = 'wandb/latest-run/files/checkpoints'
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)


    env = ModifiedDungeon(**config.env)
    fix_seed(env, config.training_params.other.random_seed)
    ppo_agent = PPO(config.model, **config.training_params.ppo)

    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    h_params = config.training_params.other

    while time_step <= h_params.max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        for t in range(h_params.max_ep_len):
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)

            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            if time_step % h_params.update_timesteps == 0:
                losses = ppo_agent.update()
                for k, v in losses.items():
                    wandb.log({k: v})

            if time_step % h_params.log_freq == 0:
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                wandb.log({'avg reward': log_avg_reward})

                log_running_reward = 0
                log_running_episodes = 0

            if time_step % h_params.render_freq == 0:
                env = ModifiedDungeon(**config.env)
                state = env.reset()
                Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500), Image.NEAREST).save(
                    'tmp.png')

                frames = []
                ppo_agent.eval()
                for _ in range(500):
                    action = ppo_agent.select_action(state)

                    frame = Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500),
                                                                                               Image.NEAREST).quantize()
                    frames.append(frame)

                    # frame.save('tmp1.png')
                    obs, reward, done, info = env.step(action)
                    if done:
                        break

                frames[0].save(f"wandb/latest-run/logs/out.gif", save_all=True, append_images=frames[1:], loop=0, duration=1000 / 60)
                wandb.log({"trajectory": wandb.Video(f"wandb/latest-run/logs/out.gif", fps=4, format="gif")})
                ppo_agent.train()


            if time_step % h_params.print_freq == 0:

                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print(f"Episode : {i_episode} Timestep : {time_step} \t Average Reward : {print_avg_reward}")

                print_running_reward = 0
                print_running_episodes = 0

            if time_step % h_params.save_model_freq == 0:
                ckpt_path = os.path.join(ckpt_dir, f"-ts-{time_step}.ckpt")
                ppo_agent.save(ckpt_path)
                wandb.save(ckpt_path)

            if done:
                break
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1


if __name__ == "__main__":
    train()
