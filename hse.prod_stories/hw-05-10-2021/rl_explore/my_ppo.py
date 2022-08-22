import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, Union
################################## set device ##################################

print("============================================================================================")

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    device = torch.device('cpu')
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ConvDenseNet(nn.Module):
    def __init__(self, in_shape, in_channels, out_dim, conv_filters, post_fcn_hiddens, actor=True):
        super().__init__()
        conv_layers = []
        for channels, kernel, stride in conv_filters:
            padding, in_shape = same_padding(in_shape, kernel, stride)
            conv_layers.extend([nn.Conv2d(in_channels, channels, kernel, stride=stride, padding=padding), nn.ReLU()])
            in_channels = channels
        linear_layers = []
        in_dim = in_shape[0] * in_shape[1] * in_channels
        for hidden_dim in post_fcn_hiddens:
            linear_layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        linear_layers.append(nn.Linear(in_dim, out_dim))
        if actor:
            linear_layers.append(nn.Softmax(dim=-1))
        self.fcn = nn.Sequential(*conv_layers)
        self.ff = nn.Sequential(*linear_layers)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
        elif len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2)
        else:
            raise ValueError('Wrong input dimensionality')
        bs = x.shape[0]
        return self.ff(self.fcn(x).view(bs, -1))


class ActorCritic(nn.Module):
    def __init__(self, config):
        super(ActorCritic, self).__init__()

        self.actor = ConvDenseNet(**config, actor=True)
        config_critic = config.copy()
        config_critic['out_dim'] = 1
        self.critic = ConvDenseNet(**config_critic, actor=False)

    def forward(self):
        raise NotImplementedError

    def act(self, state):

        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):

        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, model_params, lr_actor, lr_critic, gamma, n_epochs, eps_clip, entropy_coeff, vf_loss_coeff):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.entropy_coeff = entropy_coeff
        self.vf_loss_coeff = vf_loss_coeff
        self.n_epochs = n_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(model_params).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(model_params).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self._eval = False

        self.MseLoss = nn.MSELoss()

    def eval(self):
        self._eval = True

    def train(self):
        self._eval = False

    def select_action(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)

        if not self._eval:
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

        return action.item()

    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        avg_critic_loss = 0
        avg_actor_loss = 0
        avg_entropy = 0
        avg_loss = 0

        # Optimize policy for n epochs
        for _ in range(self.n_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            critic_loss = -torch.min(surr1, surr2).mean()
            actor_loss = F.mse_loss(state_values, rewards)
            dist_entropy = dist_entropy.mean()
            loss = critic_loss + self.vf_loss_coeff * actor_loss - self.entropy_coeff * dist_entropy

            avg_critic_loss += critic_loss.item()
            avg_actor_loss += actor_loss.item()
            avg_entropy += dist_entropy.item()
            avg_loss += loss.item()

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        avg_actor_loss /= self.n_epochs
        avg_critic_loss /= self.n_epochs
        avg_entropy /= self.n_epochs
        avg_loss /= self.n_epochs

        return {'critic_loss': avg_critic_loss, 'actor_loss': avg_actor_loss, 'entropy': avg_entropy, 'resulting_loss': avg_loss}

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


def same_padding(in_size: Tuple[int, int], filter_size: Tuple[int, int],
                 stride_size: Union[int, Tuple[int, int]]
                 ) -> (Union[int, Tuple[int, int]], Tuple[int, int]):
    """Note: Padding is added to match TF conv2d `same` padding. See
        www.tensorflow.org/versions/r0.12/api_docs/python/nn/convolution
    Args:
        in_size (tuple): Rows (Height), Column (Width) for input
        stride_size (Union[int,Tuple[int, int]]): Rows (Height), column (Width)
            for stride. If int, height == width.
        filter_size (tuple): Rows (Height), column (Width) for filter
    Returns:
        padding (tuple): For input into torch.nn.ZeroPad2d.
        output (tuple): Output shape after padding and convolution.
    """
    in_height, in_width = in_size
    if isinstance(filter_size, int):
        filter_height, filter_width = filter_size, filter_size
    else:
        filter_height, filter_width = filter_size
    if isinstance(stride_size, (int, float)):
        stride_height, stride_width = int(stride_size), int(stride_size)
    else:
        stride_height, stride_width = int(stride_size[0]), int(stride_size[1])

    out_height = np.ceil(in_height / stride_height)
    out_width = np.ceil(in_width / stride_width)

    pad_along_height = int(
        ((out_height - 1) * stride_height + filter_height - in_height))
    pad_along_width = int(
        ((out_width - 1) * stride_width + filter_width - in_width))
    pad_top = 0 if pad_along_height == 0 else max(pad_along_height // 2, 1)
    # pad_bottom = pad_along_height - pad_top
    pad_left = 0 if pad_along_width == 0 else max(pad_along_width // 2, 1)
    # pad_right = pad_along_width - pad_left
    padding = (pad_left, pad_top)
    output = (int(out_height), int(out_width))
    return padding, output
