from loss_functions import *
import datetime
import os
import copy
from stable_marriage import *

class Trainer():
    """
    Trainer class
    """

    def __init__(self,  device='cpu'):
        
        self.all_train_losses = []
        self.all_dev_losses = []
        self.all_test_losses = [] 
        self.device = device
        self.time_stamp = self.get_time_stamp()
        self.best_performance_data = {'train_loss': 10000, 'dev_loss': 10000, 'last_epoch_saved': -1000, 'model_params_to_save': None}
    
    def reset(self):
        """
        Reset the losses
        """

        self.all_train_losses = []
        self.all_dev_losses = []
        self.all_test_losses = []
    
    def get_time_stamp(self):

        return int(datetime.datetime.now().timestamp())

    def get_year_month_day(self):
        """"
        Get current date in year_month_day format
        """

        ct = datetime.datetime.now()
        return f"{ct.year}_{ct.month:02d}_{ct.day:02d}"
    
    def get_proposal_index_from_matrix(self, proposal_matrix, dim=3):
        """
        For each row (corresponding to a man), return the index of the entry with the highest value
        We first concatenate the matrix with a matrix of 1s - sum of the matrix along the last dimension, representing the case where no proposal is made

        Args:
        proposal_matrix: tensor in which the last 2 dimensions are of length m and w, respectively
        """

        proposals_and_no_proposals = torch.cat([proposal_matrix, 1 - torch.sum(proposal_matrix, dim=dim).unsqueeze(dim)], dim=dim)    
        return torch.argmax(proposals_and_no_proposals, dim=dim)

    def train(self, epochs, loss_function, model, data_loaders, optimizer, feature_keys, label_keys, trainer_params):
        """
        Train a parameterized policy
        """

        for epoch in range(epochs): # Make multiple passes through the dataset
            
            # Do one epoch of training, including updating the model parameters
            average_train_loss = self.do_one_epoch(
                optimizer, 
                data_loaders['train'], 
                loss_function, 
                model, 
                feature_keys,
                label_keys ,
                train=True, 
                epoch=epoch, 
                )
            self.all_train_losses.append(average_train_loss)

            # Do one epoch of dev, including updating the model parameters
            average_dev_loss = self.do_one_epoch(
                optimizer, 
                data_loaders['dev'], 
                loss_function, 
                model, 
                feature_keys,
                label_keys ,
                train=False, 
                )
            self.all_dev_losses.append(average_dev_loss)
            
            print(f'epoch: {epoch + 1}')
            print(f'Average train loss: {average_train_loss}')
            print(f'Average dev loss: {average_dev_loss}')

            # update best model parameters if it achieves best performance so far, and save the model (if minimum number of epochs between saves has passed)
            self.update_best_params_and_save(epoch, average_train_loss, average_dev_loss, trainer_params, model, optimizer)
    
    def test(self, loss_function, model, data_loaders, optimizer, feature_keys, label_keys):


        average_test_loss = self.do_one_epoch(
                optimizer, 
                data_loaders['test'], 
                loss_function, 
                model, 
                feature_keys,
                label_keys ,
                train=False,
                test=True, 
                )

        print(f'Average test loss: {average_test_loss}')
        
        return average_test_loss

    def do_one_epoch(self, optimizer, data_loader, loss, model, feature_keys, label_keys, train=True, test=False, epoch=0):
        """
        Do one epoch of training or testing
        """
        
        epoch_loss = 0
        total_samples = len(data_loader.dataset)
        correct_proposal_predictions = 0
        correct_matching_predictions = 0
        correct_final_matching_predictions = 0
        correct_complete_final_matching_predictions = 0
        total_predictions = 0
        total_final_predictions = 0
        total_final_samples = 0
        is_stable_counter = 0
        finished_counter = 0

        for i, data_batch in enumerate(data_loader):  # Loop through batches of data
            all_predicted_proposals = []
            all_predicted_matchings = []
            data_batch = self.move_batch_to_device(data_batch)
            feature_batch = {k: data_batch[k] for k in feature_keys}
            target_batch = {label: data_batch[label] for label in label_keys}
            max_t = feature_batch[feature_keys[0]].shape[1]

            # as we want the algorithm to "flow" through the time horizon, we will keep track of the simulated proposals (i.e. matrix of all proposals done so far,
            # with a 1 representing that the i-th man already proposed to the j-th woman) and the simulated matching (i.e. the matching matrix at after 
            # t iterations of the algorithm)
            simulated_proposals = feature_batch['current_proposal_matrix'][:, 0]
            simulated_matching = feature_batch['current_matching'][:, 0]
            
            if train:
                # Zero-out the gradient
                optimizer.zero_grad()

            for t in range(max_t):
                # Forward pass

                ############ proposal step ############
                feats_to_feed = {k: v[:, t] for k, v in feature_batch.items()}
                feats_to_feed.update({'test': test})
                # feats_to_feed['current_proposal_matrix'] = simulated_proposals

                # In theory, we could apply teacher forcing, with which we would replace the current proposal matrix or current matching given by
                # Our neural networks with the "correct" values of proposal matrix or matching matrix.
                # prob is the probability of NOT resetting the algorithm
                prob = 1.0
                # In early, experiment, teacher forcing did not work well, so we will not use it. otherwise, we would uncomment the following lines
                # if train:  # probability of NOT resetting the algorithm
                #     prob = 1 - 0.995**epoch # probability of NOT resetting the algorithm increases exponentially with the number of epochs

                # indexes is a tensor of shape (batch_size, 1, 1) with values 0 or 1, where 1 means that we will use the simulated_proposals, 
                # and 0 means that we will use the predicted proposals
                indexes =(torch.rand(simulated_proposals.shape[0]) < prob).to(self.device).unsqueeze(1).unsqueeze(2)
                # indexes =(torch.rand(simulated_proposals.shape[0], 1, 1) < prob).to(self.device)
                simulated_proposals = self.random_teacher_forcing(simulated_proposals, feature_batch['current_proposal_matrix'][:, t], indexes)
                simulated_matching = self.random_teacher_forcing(simulated_matching, feature_batch['current_matching'][:, t], indexes)
                feats_to_feed['current_proposal_matrix'] = simulated_proposals
                #create a new tensor such that, for each i, the i-th element corresponds to the i-th row of the matrix feats_to_feed['current_proposal_matrix'] and otherwise it corresponds to simulated_proposals
                feats_to_feed['current_matching'] = simulated_matching
                pred_proposal = model.forward(feats_to_feed, proposals=True)              
                simulated_proposals = simulated_proposals + pred_proposal
                # C = torch.where(torch.rand(A.shape[0], 1, 1) < 0.5, A[0].unsqueeze(1), B[0].unsqueeze(1))
                all_predicted_proposals.append(pred_proposal)

                ############ proposal step ############
                feats_to_feed = {k: v[:, t] for k, v in feature_batch.items()}
                feats_to_feed.update({'test': test})
                # feats_to_feed['new_proposals_matrix'] = target_batch['new_proposals_matrix'][:, t]
                feats_to_feed['new_proposals_matrix'] = self.random_teacher_forcing(pred_proposal, target_batch['new_proposals_matrix'][:, t], indexes)
                # feats_to_feed['new_proposals_matrix'] = pred_proposal
                feats_to_feed['current_matching'] = simulated_matching
                pred_matching = model.forward(feats_to_feed, proposals=False)
                simulated_matching = pred_matching
                all_predicted_matchings.append(pred_matching)

                # print(f"simulated_proposals: {simulated_proposals[0].round()}")
                # print()
                # batch_loss = loss(pred, {label: v[:, t] for label, v in target_batch.items()})
            # Forward pass
            # pred = model.forward(feature_batch)
            stacked_predicted_proposals = torch.stack(all_predicted_proposals, dim=1)
            stacked_predicted_matchings = torch.stack(all_predicted_matchings, dim=1)
            # print device of stacked_predicted_proposals
            # print(f"stacked_predicted_proposals.device: {stacked_predicted_proposals.device}")


            # print(f"stacked_predictions: {stacked_predictions.shape}")
            batch_loss = loss(stacked_predicted_proposals, target_batch['new_proposals_matrix'], target_batch['not_finished'], {'m_preferences': feature_batch['m_preferences']}, proposal=True) + \
                loss(stacked_predicted_matchings, target_batch['next_matching_matrix'], target_batch['not_finished'], {'w_preferences': feature_batch['w_preferences']}, proposal=False)
            
            # batch_loss = loss(pred, target_batch['new_proposals_matrix'])  # Rewards from period 0
            epoch_loss += batch_loss  # Rewards from period 0
            total_samples += stacked_predicted_proposals.shape[0]
            
            # print(f"batch_loss: {batch_loss}")
            # print(f"epoch_loss: {epoch_loss}")
            mean_loss = batch_loss/(target_batch['not_finished'].sum() * stacked_predicted_proposals.shape[2]* stacked_predicted_proposals.shape[3])
            # print(f"target_batch['not_finished'].sum(): {target_batch['not_finished'].sum()}")
            # print(f"mean_loss: {mean_loss}")
            # mean_loss = batch_loss/(stacked_predicted_proposals.shape[0] * stacked_predicted_proposals.shape[1] * stacked_predicted_proposals.shape[2]* stacked_predicted_proposals.shape[3])
            
            # Backward pass (to calculate gradient) and take gradient step
            if train:
                mean_loss.backward()
                optimizer.step()
            
            if test:
                # number of times the algorithm has finished (i.e. the number of times the algorithm has reached the end of the time horizon)
                num_alg_finished = (1 - target_batch['not_finished']).sum().item()
                num_alg_not_finished = (target_batch['not_finished']).sum().item()
                print(f"num_alg_finished: {num_alg_finished}")
                print(f"num_alg_not_finished: {num_alg_not_finished}")
                proposal_indices = self.get_proposal_index_from_matrix(stacked_predicted_proposals)
                target_proposal_indices = self.get_proposal_index_from_matrix(target_batch['new_proposals_matrix'])
                # print(f"target_batch['not_finished']: {target_batch['not_finished'][0]}")
                print(f'm_preferences: {feature_batch["m_preferences"][0][0]}')               
                print(f'w_preferences: {feature_batch["w_preferences"][0][0]}')               
                print(f'proposal_indices: {proposal_indices[0]}')               
                print(f'target_proposal_indices: {target_proposal_indices[0]}')               
                # count the number of correct predictions
                expanded_not_finished = target_batch['not_finished'].unsqueeze(2).repeat(1, 1, proposal_indices.shape[2])
                correct_proposal_predictions += torch.sum((proposal_indices == target_proposal_indices)*expanded_not_finished).item()
                print(f'proposal_indices == target_proposal_indices: {(proposal_indices == target_proposal_indices)[0].int()}')
                total_predictions += proposal_indices.shape[0] * proposal_indices.shape[1] * proposal_indices.shape[2] - num_alg_finished* proposal_indices.shape[2]

                matching_indices = self.get_proposal_index_from_matrix(stacked_predicted_matchings)
                target_matching_indices = self.get_proposal_index_from_matrix(target_batch['next_matching_matrix'])               
                # count the number of correct predictions
                correct_matching_predictions += torch.sum((matching_indices == target_matching_indices)*expanded_not_finished).item()
                print(f'matching_indices: {matching_indices[0]}')               
                print(f'target_matching_indices: {target_matching_indices[0]}')               
                print(f'matching_indices == target_matching_indices: {(matching_indices == target_matching_indices)[0].int()}')
                print()

                round_alg_finished = torch.clip(target_batch['not_finished'].sum(dim=1) - 1, max=len(target_batch['not_finished'][0])).to(torch.int64)
                # for each row in matching_indices, gather the row corresponding to round_alg_finished
                # print(f'Round alg finished: {round_alg_finished}')
                # matching_indices_last = torch.gather(matching_indices, 1, round_alg_finished)
                # print(f'matching_indices: {matching_indices[0]}')
                # print(f'target_matching_indices: {target_matching_indices[0]}')
                matching_indices_last = torch.gather(matching_indices, 1, round_alg_finished.unsqueeze(1).unsqueeze(2).repeat(1, 1, matching_indices.shape[2])).squeeze(1)
                target_matching_indices_last = torch.gather(target_matching_indices, 1, round_alg_finished.unsqueeze(1).unsqueeze(2).repeat(1, 1, matching_indices.shape[2])).squeeze(1)

                correct_final_matching_predictions += torch.sum(matching_indices_last == target_matching_indices_last).item()
                correct_complete_final_matching_predictions += (torch.sum(matching_indices_last == target_matching_indices_last, dim=1) == matching_indices_last.shape[1]).sum()
                # print(f'correct_final_matching_predictions vector: {torch.sum(matching_indices_last == target_matching_indices_last, dim=1) == matching_indices_last.shape[1]}')
                # print(f'correct_final_matching_predictions: {correct_final_matching_predictions}')
                total_final_predictions += matching_indices_last.shape[0]*matching_indices_last.shape[1]
                total_final_samples += matching_indices_last.shape[0]

                matching_dict_last = {i: matching_indices_last[i] for i in range(matching_indices_last.shape[0])}

                all_m_preferences = feature_batch['m_preferences']
                all_w_preferences = feature_batch['w_preferences']
                for sample in range(len(all_m_preferences)):
                    if target_batch["not_finished"][:, -1][sample] == 0:
                        finished_counter += 1
                        # second index corresponds to time index, and preferences are the same across time, so we only consider t=0
                        m_preferences = all_m_preferences[sample, 0]
                        w_preferences = all_w_preferences[sample, 0]

                        # for each row i in m_preferences, return a list of indices j in descending orders by the value m_preferences[i, j]
                        guyprefers = {i: m_preferences[i].argsort(descending=True).tolist() for i in range(m_preferences.shape[0])}
                        galprefers = {i: w_preferences[i].argsort(descending=True).tolist() for i in range(w_preferences.shape[0])}
                        this_matching = {i: val for i, val in enumerate(matching_dict_last[sample].tolist()) if val != m_preferences.shape[1]}
                        marriage_model = MarriageModel(guyprefers, galprefers)
                        if marriage_model.is_stable(this_matching) == True:
                            is_stable_counter += 1

                # print(f'matching_indices_last: {matching_indices_last[0]}')
                # print(f'matching_indices_last: {matching_indices_last.shape}')
        
        if test:
            proposal_accuracy = correct_proposal_predictions/total_predictions
            matching_accuracy = correct_matching_predictions/total_predictions
            final_matching_accuracy = correct_final_matching_predictions/total_final_predictions
            complete_final_matching_accuracy = correct_complete_final_matching_predictions/total_final_samples
            print(f"correct_final_matching_predictions: {correct_final_matching_predictions}")
            print(f"correct_complete_final_matching_predictions: {correct_complete_final_matching_predictions}")
            print(f"proposal_accuracy: {proposal_accuracy}")
            print(f"matching_accuracy: {matching_accuracy}")
            print(f"final_matching_accuracy: {final_matching_accuracy}")
            print(f"complete_final_matching_accuracy: {complete_final_matching_accuracy}")
            print(f'is_stable_ratio: {is_stable_counter/(finished_counter)}')
        del mean_loss, loss, batch_loss, pred_proposal, pred_matching, stacked_predicted_proposals, stacked_predicted_matchings
        # assert False
        return (epoch_loss/(total_samples*feature_batch['current_proposal_matrix'][:, 0].shape[1])).detach()
        # return epoch_loss/(total_samples* pred.shape[1] * pred.shape[2])
    
    def move_batch_to_device(self, data_batch):
        """
        Move a batch of data to the device (CPU or GPU)
        """

        return {k: v.to(self.device) for k, v in data_batch.items()}
    
    def create_folder_if_not_exists(self, folder):
        """
        Create a directory in the corresponding file, if it does not already exist
        """

        if not os.path.isdir(folder):
            os.mkdir(folder)
    
    def create_many_folders_if_not_exist_and_return_path(self, base_dir, intermediate_folder_strings):
        """
        Create a directory in the corresponding file for each file in intermediate_folder_strings, if it does not already exist
        """

        path = base_dir
        for string in intermediate_folder_strings:
            path += f"/{string}"
            self.create_folder_if_not_exists(path)
        return path
    
    def load_model(self, model, optimizer, model_path):
        """
        Load a saved model
        """

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.all_train_losses = checkpoint['all_train_losses']
        self.all_dev_losses = checkpoint['all_dev_losses']
        self.all_test_losses = checkpoint['all_test_losses']
        return model, optimizer
    
    def save_model(self, epoch, model, optimizer, trainer_params):

        path = self.create_many_folders_if_not_exist_and_return_path(base_dir=trainer_params['base_dir'], 
                                                                     intermediate_folder_strings=trainer_params['save_model_folders']
                                                                     )
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    # 'model_state_dict': self.best_performance_data['model_params_to_save'],
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_train_loss': self.best_performance_data['train_loss'],
                    'best_train_loss': self.best_performance_data['dev_loss'],
                    'best_dev_loss': self.all_train_losses,
                    'all_train_losses': self.all_train_losses,
                    'all_dev_losses': self.all_dev_losses,
                    'all_test_losses': self.all_test_losses,
                    }, 
                    f"{path}/{self.time_stamp}.pt"
                    )
    
    def update_best_params_and_save(self, epoch, train_loss, dev_loss, trainer_params, model, optimizer):
        """
        Update best model parameters if it achieves best performance so far, and save the model
        """

        data_for_compare = {'train_loss': train_loss, 'dev_loss': dev_loss}
        if data_for_compare[trainer_params['choose_best_model_on']] < self.best_performance_data[trainer_params['choose_best_model_on']]:  
            self.best_performance_data['train_loss'] = train_loss
            self.best_performance_data['dev_loss'] = dev_loss
            if model:
                self.best_performance_data['model_params_to_save'] = copy.deepcopy(model.state_dict())
            self.best_performance_data['update'] = True

        if trainer_params['save_model']:
            if self.best_performance_data['last_epoch_saved'] + trainer_params['epochs_between_save'] <= epoch and self.best_performance_data['update']:
                self.best_performance_data['last_epoch_saved'] = epoch
                self.best_performance_data['update'] = False
                print(f'Saving')
                self.save_model(epoch, model, optimizer, trainer_params)
    
    def random_teacher_forcing(self, A, B, indexes):
        """
        For each row, with probability prob, return the corresponding element of A, otherwise return the corresponding element of B
        """

        return torch.where(indexes, A, B)
