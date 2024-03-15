import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class Actor(nn.Module):
    def __init__(self,hidden_size: int, hidden_layers: int, actor_lr: float, activation: str,
                input_dim: int, output_dim: int, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.device = device

        # layer list
        self.layers = nn.ModuleList()

        # Linear input layer data dimension input_dim and hidden_size output dimension
        self.layers.append(nn.Linear(input_dim, hidden_size))

        # fill in the rest of the layers
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # one output layer for the mean and one for  the log_std
        output_dim =1 
        self.mean_output = nn.Linear(hidden_size, output_dim)
        self.log_std_output = nn.Linear(hidden_size, output_dim)

        # store the activation for the forward function. using my own function
        self.activation = self.get_activation(activation)

    def get_activation(self, activation: str):
        """
        checks if valid activation function was given in initialization
        and also converts it to a pytorch functional
        """
        if activation == 'relu':
            return nn.functional.relu
        else:
            raise ValueError("Unsupported activation function")
        
    def forward(self, s: torch.Tensor):
        for layer in self.layers:
            s = self.activation(layer(s))
        mean = self.mean_output(s)
        log_std = self.log_std_output(s)

        return mean,log_std
    
    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
    
    def get_action_and_log_prob(self, state, deterministic: bool):
        epsilon = 1e-6
        mean, raw_log_std = self.forward(state)
        log_std = self.clamp_log_std(raw_log_std)
        std = log_std.exp()

        if deterministic:
            action = torch.tanh(mean)
            log_prob = Normal(mean, torch.tensor(1e-3)).log_prob(action)
        else:
            normal = Normal(0,1)
            raw_action = normal.sample().to(self.device)
            action = torch.tanh(mean+std*raw_action)
            log_prob = Normal(mean,std).log_prob(mean+std*raw_action) - torch.log(1-action.pow(2) + epsilon)
        return action, log_prob

        



class Critic(nn.Module):
    def __init__(self,hidden_size: int, hidden_layers: int, actor_lr: float, activation: str,
                input_dim: int, output_dim: int, device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()

        # layer list
        self.layers = nn.ModuleList()

        # Linear input layer data dimension input_dim and hidden_size output dimension
        self.layers.append(nn.Linear(input_dim, hidden_size))

        # fill in the rest of the layers
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # one output layer
        self.output_layer = nn.Linear(hidden_size, output_dim)

        # store the activation for the forward function. using my own function
        self.activation = self.get_activation(activation)

    def get_activation(self, activation: str):
        """
        checks if valid activation function was given in initialization
        and also converts it to a pytorch functional
        """
        if activation == 'relu':
            return nn.functional.relu
        else:
            raise ValueError("Unsupported activation function")
        
    def forward(self, s: torch.Tensor, a: torch.Tensor):
        x = torch.cat([s, a], 1)
        for layer in self.layers:
            x = self.activation(layer(x))
        output = self.output_layer(x)

        return output



class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.

    Helper functions, not necessary
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, activation: str):
        super(NeuralNetwork, self).__init__()

        # TODO: Implement this function which should define a neural network 
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        pass
    


    def get_action_and_log_prob(self, state: torch.Tensor, 
                                deterministic: bool) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        action , log_prob = torch.zeros(state.shape[0]), torch.ones(state.shape[0])
        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.
        assert action.shape == (state.shape[0], self.action_dim) and \
            log_prob.shape == (state.shape[0], self.action_dim), 'Incorrect shape for action or log_prob.'
        return action, log_prob




class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        
        self.setup_agent()

    def setup_agent(self):
        # TODO: Setup off-policy agent with policy and critic classes. 
        # Feel free to instantiate any other parameters you feel you might need.   
        hidden_size = 128
        hidden_layers = 2
        lr = 8e-3
        temp_initial = 1.0
        temp_lr = 8e-3


        self.q_net1 = Critic(hidden_size,hidden_layers,
                             lr, 'relu', self.state_dim+self.action_dim,
                             self.action_dim, self.device)
        self.q_net2 = Critic(hidden_size,hidden_layers,
                             lr, 'relu', self.state_dim+self.action_dim,
                             self.action_dim, self.device)
        
        self.target_q_net1 = Critic(hidden_size,hidden_layers,
                             lr, 'relu', self.state_dim+self.action_dim,
                             self.action_dim, self.device)
        self.target_q_net2 = Critic(hidden_size,hidden_layers,
                             lr, 'relu', self.state_dim+self.action_dim,
                             self.action_dim, self.device)
        
        self.policy_net = Actor(hidden_size,hidden_layers,
                                lr, 'relu',self.state_dim,
                                self.action_dim, self.device)
        
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(param.data)
        
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(),lr=lr)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(),lr=lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(),lr=lr)

        self.alpha = TrainableParameter(init_param=temp_initial, lr_param=temp_lr,
                                        train_param=True,device=self.device)
        
        self.entropy_target = -1*self.action_dim

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
        action = np.random.uniform(-1, 1, (1,))
        state_tensor = torch.from_numpy(s)
        action, _ = self.policy_net.get_action_and_log_prob(state_tensor, not train)

        action = action.detach().cpu().numpy()

        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray' 
        return action

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork, 
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        # TODO: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)

        train = True
        gamma = 0.99
        tau = 5e-3

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch

        """
        update for the Q functions
        different steps
            sample actions from next state batch
            compute values of Qi function
            compute target values of Qi function
            compute loss
            optimization
        """
        # sample actions
        next_sampled_actions, next_log_prob = self.policy_net.get_action_and_log_prob(s_prime_batch, not train)

        # compute values Qi
        q1_value = self.q_net1(s_batch,a_batch)
        q2_value = self.q_net2(s_batch,a_batch)

        # compute target values Qi
        target_value_1 = self.target_q_net1(s_prime_batch,next_sampled_actions)
        target_value_2 = self.target_q_net2(s_prime_batch,next_sampled_actions)

        # functions containing targets
        target_func_1 = r_batch + gamma*(target_value_1 - self.alpha.get_param()*next_log_prob)
        target_func_2 = r_batch + gamma*(target_value_2 - self.alpha.get_param()*next_log_prob)

        # compute the loss
        criterion_1 = nn.MSELoss()
        criterion_2 = nn.MSELoss()

        loss_q1 = criterion_1(q1_value,target_func_1.detach())
        loss_q2 = criterion_2(q2_value,target_func_2.detach())

        # optimization
        self.q1_optimizer.zero_grad()
        loss_q1.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.zero_grad()
        loss_q2.backward()
        self.q2_optimizer.step()

        """
        now we want to update the policy weights
            sample actions and log_prob from current states
            compute the min Qi value
            compute loss
            optimize
        """
        # sample actions
        sampled_actions, log_prob = self.policy_net.get_action_and_log_prob(s_batch, not train)
        
        # compute min Qi
        min_q_value = torch.min(self.q_net1(s_batch,sampled_actions),self.target_q_net2(s_batch,sampled_actions))

        # compute loss
        policy_loss = (self.alpha.get_param()*log_prob - min_q_value).mean()

        # optimize
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()


        """
        now i want to also update the temperature parameter
            entropy can be used from previously computed log_pron
            create target entropy tensor
            compute temperature loss
            optimize
        """
        # target entropy tensor
        entropy_target_tensor = torch.full_like(log_prob,self.entropy_target)

        # loss function
        temperature_loss = (-self.alpha.get_log_param())*(log_prob.detach()+entropy_target_tensor)
        self.run_gradient_update_step(self.alpha,temperature_loss)

        """
        final step is to perform target updates
        """
        self.critic_target_update(self.q_net1,self.target_q_net1,
                                  tau=tau,soft_update=True)
        
        self.critic_target_update(self.q_net2,self.target_q_net2,
                                  tau=tau, soft_update=True)

        # TODO: Implement Critic(s) update here.

        # TODO: Implement Policy update here


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = True
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")
    
    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()
