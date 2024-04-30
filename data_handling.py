from torch.utils.data import Dataset, DataLoader
import torch
from collections import defaultdict
from stable_marriage import *
import random

class MyDataset(Dataset):

    def __init__(self, num_samples, data):
        self.data = data
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
        # return {k: v[idx] for k, v in self.data.items()}

class DatasetCreator():
    
    def __init__(self):
        pass
    
    def create_dataset(self, num_samples, m, w, feature_keys, label_key):

        complete_dataset = {}
        max_length = 0
        min_length = 1000
        sum_length = 0
        larger_than = 0
        smaller_than = 0
        max_len = 15
        # complete_dataset = defaultdict(list)

        for i in range(num_samples):
        # while len(complete_dataset['w_preferences']) < num_samples:
            dataset = self.get_marriage_data(m, w)
            # for key in feature_keys + [label_key]:
            # # for key in dataset.keys():
            #     complete_dataset[key] += dataset[key]
            complete_dataset[i] = dataset
            max_length = max(max_length, len(dataset['current_matching']))
            min_length = min(min_length, len(dataset['current_matching']))
            sum_length += len(dataset['current_matching'])
            # if len(dataset['current_matching']) > 15:
            #     larger_than += 1
            # if len(dataset['current_matching']) < 5:
            #     smaller_than += 1
            

        # print(f'max_length: {max_length}')
        # print(f'min_length: {min_length}')
        # print(f'avg_length: {sum_length/num_samples}')
        # print(f'larger_than: {larger_than/num_samples}')
        # print(f'smaller_than: {smaller_than/num_samples}')
        # assert False
        # print(f'complete_dataset[10]: {complete_dataset[10]["current_matching"]}')
        # print(f'complete_dataset[10]: {len(complete_dataset[10]["current_matching"])}')
        
        # print(f'complete_dataset[10].keys(): {complete_dataset[10].keys()}')
        # lens = {key: len(val) for key, val in complete_dataset[10].items()}
        # print(f'len complete_dataset[10].keys(): {lens}')

        # turn every list of every value type of every sample to a torch tensor
        # As each sample of the algorithm might have a different time length, we are going to truncate that algorithms
        # to the maximum time length (currently 15 periods). For samples with a length of less than 15, we are going to
        # pad them with zeros.
        complete_dataset = {
            i: {
            key: torch.stack([torch.tensor(x).float() for x in val[: max_len]], dim=0) if len(val) == max_len 
            else torch.stack([torch.tensor(x) for x in val[: max_len]] + [torch.zeros_like(torch.tensor(val[0])) for _ in range(max_len - len(val))], dim=0).float()
            for key, val in dataset.items() } 
            for i, dataset in complete_dataset.items()
            }
        # print(f'complete_dataset[10]["m_preferences"]: {complete_dataset[10]["m_preferences"][0]}')
        # print(f'complete_dataset[10]["w_preferences"]: {complete_dataset[10]["w_preferences"][0]}')
        # print(f'complete_dataset[10]["new_proposals_matrix"]: {complete_dataset[10]["new_proposals_matrix"]}')
        # print(f'complete_dataset[10]["next_matching_matrix"]: {complete_dataset[10]["next_matching_matrix"]}')
        # print(f'not_finished[10]["current_matching"]: {complete_dataset[10]["not_finished"]}')
        # assert False
        # # transpose dims 1 and 2 for 'w_preferences' on each i of complete_dataset
        # complete_dataset = {
        #     i: {
        #     key: val.transpose(1, 2) if key == 'w_preferences' else val
        #     for key, val in dataset.items() }
        #     for i, dataset in complete_dataset.items()
        #     }
        
        # print(f'current_matching: {complete_dataset[10]["current_matching"]}')
        # print(f'current_matching: {complete_dataset[10]["current_matching"].shape}')
        # for key in complete_dataset.keys():
            # complete_dataset[key] = [torch.tensor(x).float() for x in complete_dataset[key]]

        # print(len(complete_dataset['w_preferences']))
        return MyDataset(num_samples, complete_dataset)
        # return MyDataset(num_samples, {k: v[: num_samples] for k, v in complete_dataset.items()})
    
    def get_marriage_data(self, m, w):

        guys = list(np.arange(m))
        gals = list(np.arange(w)) 
        guyprefers = {g: random.sample(gals, len(gals)) for g in guys}
        galprefers = {g: random.sample(guys, len(guys)) for g in gals}

        # print(f'guyprefers: {guyprefers}')
        # print(f'galprefers: {galprefers}')

        model = MarriageModel(guyprefers, galprefers)
        mu, dataset = model.Deferred_Acceptance(print_tentative_matchings=True)

        return dataset