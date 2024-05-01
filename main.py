# from stable_marriage import *
import numpy as np
import random
from collections import defaultdict
from data_handling import *
from neural_networks import *
import yaml
from loss_functions import *
from trainer import *

dataset_creator = DatasetCreator()
# device = "cuda:0"
assert  torch.cuda.is_available()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'device: {device}')


# num_samples = {'train': 100, 'dev': 100, 'test': 100}
num_samples = {'train': 100, 'dev': 100, 'test': 10000}
# num_samples = {'train': 2**15, 'dev': 1000, 'test': 100}
# to_train, to_test = True, False
to_train, to_test = False, True
# num_samples = {'train': 100000, 'dev': 10000, 'test': 10000}
m, w = 10, 10  # number of men and women
save_additional_string = 'both_constraints'

# keys corresponding to features and labels/targets
feature_keys = ['w_preferences', 'm_preferences', 'current_matching', 'current_proposal_matrix']
label_key = ['new_proposals_matrix', 'next_matching_matrix', 'not_finished']
datasets = {k: dataset_creator.create_dataset(samples, m, w, feature_keys, label_key) for k, samples in num_samples.items()}

dataloaders = {'train': DataLoader(datasets['train'], batch_size=2**12, shuffle=True), 
               'dev': DataLoader(datasets['dev'], batch_size=2**12, shuffle=False), 
               'test': DataLoader(datasets['test'], batch_size=2**12, shuffle=False)}

# config file from which to fetch the hyperparameters
config_net_file = 'config_files/nets/encoder_symmetry_aware.yml'
# config_net_file = 'config_files/nets/symmetry_aware.yml'
# config_net_file = 'config_files/nets/vanilla.yml'

with open(config_net_file, 'r') as file:
    config_net = yaml.safe_load(file)

hyperparams_keys = 'trainer_params', 'optimizer_params', 'nn_params'
trainer_params, optimizer_params, nn_params = [config_net[key] for key in hyperparams_keys]

neural_net_creator = NeuralNetworkCreator
model = neural_net_creator().create_neural_network(nn_params=nn_params, problem_params={'m': m, 'w': w}, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss = WeightedL2Loss()
# loss = L2Loss()

trainer = Trainer(device=device)
trainer_params['base_dir'] = 'saved_models'
trainer_params['save_model_folders'] = [trainer.get_year_month_day(), nn_params['name']]
if save_additional_string != '':
    trainer_params['save_model_folders'].append(save_additional_string)
epochs = 3000

if trainer_params['load_previous_model']:
    print(f'Loading model from {trainer_params["load_model_path"]}')
    model, optimizer = trainer.load_model(model, optimizer, trainer_params['load_model_path'])

if to_train:
    trainer.train(
        epochs,
        loss, 
        model, 
        dataloaders, 
        optimizer, 
        feature_keys,
        label_key, 
        trainer_params=trainer_params, 
        )
    
if to_test:
    trainer.test(
        loss, 
        model, 
        dataloaders, 
        optimizer,
        feature_keys, 
        label_key, 
        )

