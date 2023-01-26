import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import os
import numpy as np
import time


class Net:
    """
    This class allows to compute the main methods to fit a model and predict output on testing set
    """

    def __init__(self, model, lr: float = 0.001, opt: str = 'Adam', wandb=None, save: bool = False,
                 name: str = None, verbose: bool = False, hyperopt: bool = False, seed: int = None):
        self.wandb = wandb
        self.save = save
        self.verbose = verbose
        self.hyperopt = hyperopt
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.opt = opt
        self.epochs = 1
        self.optimizer = None
        self.train_loss = None
        self.val_loss = None
        self.path = None
        self.__set_seed()
        self.__instantiate_model(model)
        self.__instantiate_save(name)
        self.__instantiate_optimizer()

    def __set_seed(self):
        torch.manual_seed(self.seed)

    def __instantiate_model(self, model):
        self.model = Utils.to_device(model, self.device)

    def __instantiate_save(self, name):
        if self.save:
            base = os.path.dirname(os.path.abspath(__file__))
            model = os.path.join(base, "models")
            assert (name is not None)
            self.path = os.path.join(model, name)
            if not os.path.exists(self.path):
                os.makedirs(self.path)

    def __instantiate_optimizer(self):
        if self.opt == 'Adam':
            self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        elif self.opt == 'SGD':
            self.optimizer = SGD(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f'{self.optimizer} has not been implemented')

    @staticmethod
    def __transf(state: np.ndarray, target: np.ndarray = None, mask: np.ndarray = None):
        state = torch.tensor(state, dtype=torch.float32)
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
        if target is not None:
            target = torch.tensor(target, dtype=torch.float32)
            if mask is not None:
                mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
                return state, target, mask
            else:
                return state, target
        return state

    def __train_model(self, state: np.ndarray, target: torch.tensor, mask: np.ndarray=None):
        # set the model in training mode
        self.model.train()
        # send input to device
        if mask is None:
            state, target = self.__transf(state, target)
            state, target = Utils.to_device((state, target), self.device)
        else:
            state, target, mask = self.__transf(state, target, mask)
            state, target, mask = Utils.to_device((state, target, mask), self.device)
        # zero out previous accumulated gradients
        self.optimizer.zero_grad() #not needed as no batch
        # perform forward pass and calculate accuracy + loss
        all_Q_values = self.model(state)
        Q_values = all_Q_values if mask is None else torch.sum(all_Q_values * mask, dim=2)[0]
        loss = self.criterion(Q_values, target)
        # perform backpropagation and update model parameters
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def __evaluate_model(self, state: np.ndarray, idmax: bool=None):
        # set the model in eval mode
        self.model.eval()
        # send input to device
        state = self.__transf(state)
        state = Utils.to_device(state, self.device)
        output = self.model(state)
        if idmax: output = Utils.argmax(output)
        return output

    def __compute_verbose_train(self, epoch, start_time, train_loss):
        print("Epoch [{}] took {:.2f}s | train_loss: {:.4f}".format(epoch, time.time() - start_time, train_loss))

    def fit(self, state: np.ndarray, target: torch.tensor, mask: np.ndarray = None):

        start_time = time.time()
        train_loss = self.__train_model(state, target, mask)

        if not self.hyperopt:
            if self.verbose:
                self.__compute_verbose_train(self.epochs, start_time, train_loss)
        else:
            pass
            # Send the current validation loss and accuration back to Tune for the hyperopt
            # Ray Tune can then use these metrics to decide which hyperparameter configuration lead to the best results.
            #tune.report(train_loss=train_loss_mean, train_acc=train_acc_mean, val_loss=val_loss_mean, val_acc=val_acc_mean)

        if self.save:
            torch.save(self.model.state_dict(), f'{self.path}/model_{self.epochs}.pt')

        self.train_loss = train_loss

    def predict(self, state: np.ndarray, idmax: bool=None):
        """
        if self.best_epoch is not None:
            epoch = self.best_epoch
        else:
            epoch = Utils.find_last_epoch(self.path)
        self.model = Utils.load_model(self.model, self.path, self.device)
        """
        prediction = self.__evaluate_model(state, idmax)
        return prediction


class Utils:
    """
    This class allows to contains all the utility function needed for neural nets
    """

    @staticmethod
    def to_device(data, device):
        if isinstance(data, (list, tuple)):
            return [Utils.to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    @staticmethod
    def argmax(outputs):
        return torch.argmax(outputs).cpu().detach().item()

    @staticmethod
    def load_model(model, path, device):
        path_ = f'{path}/model_1.pt'
        if device.type == 'cpu':
            model.load_state_dict(torch.load(path_, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(path_))
        return model
