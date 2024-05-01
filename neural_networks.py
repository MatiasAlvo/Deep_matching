import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy

class MyNeuralNetwork(nn.Module):

    def __init__(self, args, device='cpu'):
        
        super().__init__() # initialize super class
        self.device = device # define device
        self.internal_counter = 0

        # Define activation functions, which will be called in forward method
        self.activation_functions = {
            'relu': nn.ReLU(), 
            'elu': nn.ELU(), 
            'tanh': nn.Tanh(),
            'softmax_1': nn.Softmax(dim=1),
            'softmax_2': nn.Softmax(dim=2),
            # 'straight_through_hardmax': StraightThroughEstimator(),
            'straight_through_hardmax': StraightThroughHardmaxLayer(),
            'hardmax': HardmaxLayer(),
            'softplus': nn.Softplus(),
            'sigmoid': nn.Sigmoid(),
            }
        
        self.layers = {} # store layers of each neural network
        
        # Create nn.ModuleDict to store multiple neural networks
        self.net = self.create_module_dict(args)
        self.args = args
    

    
    def create_module_dict(self, args):
        """
        Create a dictionary of neural networks, where each key is a neural network module (e.g. master, store, warehouse, context)
        """
        
        return nn.ModuleDict({key: 
                              self.create_sequential_net(
                                  key,
                                  args['inner_layer_activations'][key], 
                                  args['output_layer_activation'][key], 
                                  args['neurons_per_hidden_layer'][key], 
                                  args['output_sizes'][key]
                                  ) 
                                  for key in args['output_sizes']
                                  }
                                  )
    
    def create_sequential_net(self, name, inner_layer_activations, output_layer_activation, neurons_per_hidden_layer, output_size):
        """
        Create a neural network with the given parameters
        """

        # Define layers
        layers = []
        for i, output_neurons in enumerate(neurons_per_hidden_layer):
            layers.append(nn.LazyLinear(output_neurons))
            layers.append(self.activation_functions[inner_layer_activations])

        if len(neurons_per_hidden_layer) == 0:
            layers.append(nn.LazyLinear(output_size))

        # If there is at least one inner layer, then we know the last layer's shape
        # We therefore create a Linear layer in case we want to initialize it to a certain value (not possible with LazyLinear)
        else: 
            # print(f'output_size: {output_size}')
            layers.append(nn.Linear(neurons_per_hidden_layer[-1], output_size))
        
        # If output_layer_activation is not None, then we add the activation function to the last layer
        if output_layer_activation is not None:
            layers.append(self.activation_functions[output_layer_activation])
        
        self.layers[name] = layers

        # Define network as a sequence of layers
        return nn.Sequential(*layers)
    
    def forward(self, args):
        """
        Forward pass of the neural network
        """
        raise NotImplementedError
    
    def unpack_args(self, args, keys):
        """
        Unpacks arguments from a dictionary
        """
        return [args[key] for key in keys] if len(keys) > 1 else args[keys[0]]
    
    def unpack_and_concat_args(self, args, keys, flatten=True, dim=1):
        """
        Unpacks arguments from a dictionary
        Then concatenates the flatenned unpacked arguments
        """
        args_list = self.unpack_args(args, keys)
        if flatten:
            return torch.cat([x.flatten(start_dim=1) for x in args_list], dim=dim)
        else:
            return torch.cat(args_list, dim=dim)

class Vanilla(MyNeuralNetwork):

    def __init__(self, args, device='cpu'):
        super().__init__(args, device=device)        
    
    def forward(self, args):
        """
        Forward pass of the neural network
        """
        x = self.unpack_and_concat_args(args, ['w_preferences', 'm_preferences', 'current_matching', 'current_proposal_matrix'])
        x = self.net['master'](x)
        x = x.reshape(-1, self.args['m'], self.args['w'] + 1)
        x = self.activation_functions['softmax_2'](x)
        # x = x.reshape(-1, self.args['m']*self.args['w'])
        # print(f"x: {x[0][10: 20].sum()}")
        # print(f'x.shape: {x.shape}')
        return x[:, :, : -1]

class SymmetryAware(MyNeuralNetwork):
    
    def forward(self, args, proposals=True):
        """
        Forward pass of the neural network
        """

        if proposals:
            # check if type of self is EncoderSymmetryAware
            
            if isinstance(self, EncoderSymmetryAware):
                x  = self.unpack_and_concat_args(args, ['current_matching', 'current_proposal_matrix'], flatten=False, dim=2)
                # print(f'x: {x.shape}')
                # print(f"m_preferences: {args['m_preferences'].shape}")
                encoded_man_features = self.net['man_encoder'](args['m_preferences'])
                # print(f'encoded_man_features: {encoded_man_features.shape}')
                x = torch.cat([encoded_man_features, x], dim=2)
                # print(f'x: {x.shape}')
            else:
                x  = self.unpack_and_concat_args(args, ['m_preferences', 'current_matching', 'current_proposal_matrix'], flatten=False, dim=2)
            x = self.net['man'](x)
            x = self.activation_functions['softmax_2'](x)
            if args['test']:
                x = self.activation_functions['hardmax'](x).float().detach()
            x = x[:, :, : -1]
            return x
        else:
            if isinstance(self, EncoderSymmetryAware):
                current_matching, new_proposals_matrix  = self.unpack_args(args, ['current_matching', 'new_proposals_matrix'])
                # print(f'x: {x.shape}')
                # print(f"m_preferences: {args['m_preferences'].shape}")
                encoded_woman_features = self.net['woman_encoder'](args['w_preferences'])
                # print(f'encoded_man_features: {encoded_man_features.shape}')
                # x = torch.cat([encoded_woman_features, x], dim=2)
                # print(f'x: {x.shape}')
            else:
                w_preferences, current_matching, new_proposals_matrix  = self.unpack_args(args, ['w_preferences', 'current_matching', 'new_proposals_matrix'])
                encoded_woman_features = w_preferences

            x = torch.cat([encoded_woman_features] + [torch.transpose(val, 1, 2) for val in [current_matching, new_proposals_matrix]], dim=2)
            x = self.net['woman'](x)
            x = self.activation_functions['softmax_2'](x)
            if args['test']:
                # print(f'x: {x[0]}')
                x = self.activation_functions['hardmax'](x).float().detach()
            x = x[:, :, : -1].transpose(1, 2)

            return x
            # return x[:, :, : -1].transpose(1, 2)

class EncoderSymmetryAware(SymmetryAware):

    pass

class STHardmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        idxs = torch.argmax(x, dim=2)
        print(f'x: {x[0, 0]}')
        print(f'idxs: {idxs[0, 0]}')
        print(f'hardmax: {F.one_hot(idxs, num_classes=x.shape[2])[0, 0]}')
        print()
        x = F.one_hot(idxs, num_classes=x.shape[2])*x
        return x
        # return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class StraightThroughHardmaxLayer(nn.Module):

    def __init__(self):
        super(StraightThroughHardmaxLayer, self).__init__()

    def forward(self, x):
            x = STHardmaxFunction.apply(x)
            return x

class HardmaxLayer(nn.Module):

    def forward(self, x):
        idxs = torch.argmax(x, dim=2)
        x = F.one_hot(idxs, num_classes=x.shape[2])
        return x

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)
class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()
    def forward(self, x):
        x = STEFunction.apply(x)
        return x

class NeuralNetworkCreator:
    """
    Class to create neural networks
    """

    def set_default_output_size(self, module_name, problem_params):
        
        default_sizes = {
            'master': problem_params['n_stores'] + problem_params['n_warehouses'], 
            'store': 1, 
            'warehouse': 1, 
            'context': None
            }
        return default_sizes[module_name]

    def get_architecture(self, name):

        architectures = {
            'symmetry_aware': SymmetryAware, 
            'encoder_symmetry_aware': EncoderSymmetryAware, 
            # 'vanilla': Vanilla, 
            }
        return architectures[name]

    
    def create_neural_network(self, nn_params, problem_params, device='cpu'):

        nn_params_copy = deepcopy(nn_params)
        nn_params_copy.update(problem_params)

        # # If not specified in config file, set output size to default value
        # for key, val in nn_params_copy['output_sizes'].items():
        #     if val is None:
        #         nn_params_copy['output_sizes'][key] = self.set_default_output_size(key, scenario.problem_params)

        model = self.get_architecture(nn_params_copy['name'])(
            nn_params_copy, 
            device=device
            )
        
        return model.to(device)
    