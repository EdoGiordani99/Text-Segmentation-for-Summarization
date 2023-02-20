import os
import torch
import datetime

from torch import nn
from tqdm import tqdm
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from utils import compute_f1, compute_pk


class SegmentationTrainer():

    def __init__(self,
                 model: nn.Module,
                 train_dataloader: DataLoader,
                 valid_dataloader: DataLoader,
                 params: dict,
                 compute_f1: bool = False,
                 save_folder: str = 'store',
                 save_history: bool = True,
                 save_model: bool = True,
                 save_best: bool = True):

        """
        This trainer class trains your model! First initialize the class and train with Trainer.train method.
        Args:
            model (nn.Module): the model to train
            train_dataloader (Dataloader): train data loader
            valid_dataloader (Dataloader): valid data loader
            params (dict of dicts): configuration dictionary with all parameters
            compute_f1 (bool): if true, f1 score is computed during training
            save_folder (str): path where you want the logs to be saved
            save_history (bool): if true, history is saved in save_folder
            save_model (bool): if true, model is saved in save_folder
            save_best (bool): if true, model is saved only when overcomes the highest validation f1 score
        """

        self.history = {'train_loss': [],
                        'train_step_loss': [],
                        'train_f1': [],
                        'train_pk': [],
                        'valid_loss': [],
                        'valid_step_loss': [],
                        'valid_f1': [],
                        'valid_pk': [],
                        'best_epoch': 0}

        self.params = params
        train_params = self.params['train']
        self.epochs = train_params['epochs']

        # Setting Device
        if train_params['device'] == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        elif train_params['device'] == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
            print('WARNING: cuda is not available. Device is set to CPU')
        else:
            self.device = 'cpu'

        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        if train_params['optimizer'] == 'Adam':
            self.optimizer = Adam(params=self.model.parameters(),
                                  lr=train_params['learning_rate'],
                                  weight_decay=train_params['weight_decay'])
        elif train_params['optimizer'] == 'SGD':
            self.optimizer = SGD(params=self.model.parameters(),
                                 lr=train_params['learning_rate'],
                                 momentum=train_params['momentum'])
        else:
            raise ValueError('Optimizer can be only "Adam" or "SGD".')

        self.compute_f1 = compute_f1

        self.save_folder = save_folder
        self.save_history = save_history
        self.save_best = save_best
        self.save_model = save_model

    def train(self):

        """
        Returns:
        history (dict): dictionary containing 'train_loss',
          'train_step_loss', 'train_f1', 'train_pk', 'valid_loss', 'valid_step_loss',
          'valid_f1', 'valid_pk', 'best_epoch'
        """

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Creating folder to save training log + model weights
        t = datetime.datetime.now()
        daytime = str(t.year) + ':' + str(t.month) + ':' + str(t.day) + '-' + str(t.hour) + ':' + str(t.minute)

        save_path = self.save_folder + '/' + daytime

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Saving Hyperparameters used
        torch.save(self.params, save_path + '/params.pth')

        # To save the best
        best_valid_f1 = -100

        # FOR EACH EPOCH
        for epoch in range(1, self.epochs + 1):

            if epoch < 10:
                print(f'\nEpoch 0{epoch}:')
            if epoch >= 10:
                print(f'\nEpoch {epoch}:')

            # -----------------------------------------------------
            #                      TRAINING
            # -----------------------------------------------------
            self.model.train()

            train_epoch_loss = 0
            train_epoch_f1 = 0
            train_epoch_pk = 0
            t_progbar = tqdm(enumerate(self.train_dataloader),
                             total=self.train_dataloader.__len__(),
                             desc='Training: ')

            for i, train_batch in t_progbar:

                self.optimizer.zero_grad()

                # Forward Pass
                if self.compute_f1:
                    out = self.model(train_batch,
                                     device=self.device,
                                     compute_preds=True)
                else:
                    out = self.model(train_batch,
                                     device=self.device)

                # Loss Backward
                loss = out['loss']
                loss.backward()
                self.optimizer.step()

                self.history['train_step_loss'].append(loss.item())
                train_epoch_loss += loss

                # Computing F1-Score & PK-Score
                if self.compute_f1:
                    f1 = compute_f1(train_batch['labels'], out['preds'])
                    pk = compute_pk(train_batch['labels'], out['preds'])
                    train_epoch_f1 += f1
                    train_epoch_pk += pk
                    t_progbar.set_postfix({'train loss': round(loss.item(), 4),
                                           'train f1': f1,
                                           'train pk': pk})
                else:
                    t_progbar.set_postfix({'train loss': round(loss.item(), 4)})

            # Train Loss
            train_epoch_loss = train_epoch_loss.item() / len(self.train_dataloader)
            self.history['train_loss'].append(train_epoch_loss)

            # Train F1 Score
            train_epoch_f1 = train_epoch_f1 / len(self.train_dataloader)
            self.history['train_f1'].append(train_epoch_f1)

            # Train PK Score
            train_epoch_pk = train_epoch_pk / len(self.train_dataloader)
            self.history['train_pk'].append(train_epoch_pk)

            if self.compute_f1:
                t_progbar.set_postfix({'train loss': round(train_epoch_loss, 4),
                                       'train f1': train_epoch_f1,
                                       'train pk': train_epoch_pk})
            else:
                t_progbar.set_postfix({'train loss': round(train_epoch_loss, 4)})

            # -----------------------------------------------------
            #                    VALIDATING
            # -----------------------------------------------------
            self.model.eval()
            valid_epoch_loss = 0
            valid_epoch_f1 = 0
            valid_epoch_pk = 0
            v_progbar = tqdm(enumerate(self.valid_dataloader),
                             total=self.valid_dataloader.__len__(),
                             desc='Training: ')

            for i, valid_batch in v_progbar:

                # Forward Pass
                if self.compute_f1:
                    out = self.model(valid_batch,
                                     device=self.device,
                                     compute_preds=True)
                else:
                    out = self.model(valid_batch,
                                     device=self.device)

                # Loss
                loss = out['loss']
                self.history['valid_step_loss'].append(loss.item())
                valid_epoch_loss += loss

                # F1-Score
                if self.compute_f1:
                    f1 = compute_f1(valid_batch['labels'], out['preds'])
                    pk = compute_pk(valid_batch['labels'], out['preds'])
                    valid_epoch_f1 += f1
                    valid_epoch_pk += pk
                    v_progbar.set_postfix({'valid loss': round(loss.item(), 4),
                                           'valid f1': f1,
                                           'valid pk': pk})
                else:
                    v_progbar.set_postfix({'valid loss': round(loss.item(), 4)})

            # Valid Loss
            valid_epoch_loss = valid_epoch_loss.item() / len(self.valid_dataloader)
            self.history['valid_loss'].append(valid_epoch_loss)

            # Valid F1 Score
            valid_epoch_f1 = valid_epoch_f1 / len(self.valid_dataloader)
            self.history['valid_f1'].append(valid_epoch_f1)

            # Valid PK Score
            valid_epoch_pk = valid_epoch_pk / len(self.valid_dataloader)
            self.history['valid_pk'].append(valid_epoch_pk)

            if self.compute_f1:
                print(f'Train loss: {round(train_epoch_loss, 4)}\t Valid Loss: {round(valid_epoch_loss, 4)}')
                print(f'Train f1: {round(train_epoch_f1, 4)}\t Valid F1: {round(valid_epoch_f1, 4)}')
                print(f'Train pk: {round(train_epoch_pk, 4)}\t Valid pk: {round(valid_epoch_pk, 4)}')
            else:
                print(f'Train loss: {round(train_epoch_loss, 4)}\t Valid Loss: {round(valid_epoch_loss, 4)}')

            # -----------------------------------------------------
            #                       SAVING
            # ----------------------------------------------------

            # Save Paths
            model_path = save_path + '/model.pth'
            hist_path = save_path + '/history.pth'

            # Saving the Best Model
            if self.save_model and self.save_best:
                if valid_epoch_f1 > best_valid_f1:
                    print('BEST EPOCH')
                    self.history['best_epoch'] = epoch
                    torch.save(self.model.state_dict(), model_path)
                    best_valid_f1 = valid_epoch_f1

            # Saving Current Model
            elif self.save_model:
                torch.save(self.model.state_dict(), model_path)

            # History Save
            if self.save_history:
                torch.save(self.history, hist_path)

        return self.history
