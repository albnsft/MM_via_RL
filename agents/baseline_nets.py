import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import os
import numpy as np
import time


class EarlyStopping:
    """
    This class allows to :
    Stop the training when the training loss doesn't decrease anymore
    Useful to reduce the number of epoch
    """

    def __init__(
            self,
            tolerance: int = 10):

        self.tolerance = tolerance
        self.counter = 0
        self.early_stop = False
        self.best_loss = 1e5

    def __call__(self, loss):
        if round(loss, 3) >= round(self.best_loss, 3):
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0


class Net:
    """
    This class allows to compute the main methods to fit a model and predict output on testing set
    """

    def __init__(self, model, lr: float = 0.01, epochs: int = 1, opt: str = 'Adam', wandb=None, save: bool = False,
                 name: str = None, verbose: bool = False, hyperopt: bool = False):
        self.wandb = wandb
        self.save = save
        self.verbose = verbose
        self.hyperopt = hyperopt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.epochs = epochs
        self.opt = opt
        self.optimizer = None
        self.best_epoch = None
        self.train_loss = None
        self.val_loss = None
        self.path = None
        self.__instantiate_model(model)
        self.__instantiate_save(name)
        self.__instantiate_optimizer()

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
    def __transf(state: np.ndarray):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        #if target is not None: target = torch.tensor(target, dtype=torch.float32)
        return state

    def __train_model(self, state: np.ndarray, target: torch.tensor):
        # set the model in training mode
        self.model.train()
        # send input to device
        state = self.__transf(state)
        state, target = Utils.to_device((state, target), self.device)
        # zero out previous accumulated gradients
        self.optimizer.zero_grad() #not needed as no batch
        # perform forward pass and calculate accuracy + loss
        outputs = self.model(state)
        loss = self.criterion(outputs, target)
        # perform backpropagation and update model parameters
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def __evaluate_model(self, state: np.ndarray, item: bool=None):
        # set the model in eval mode
        self.model.eval()
        # send input to device
        state = self.__transf(state)
        state = Utils.to_device(state, self.device)
        output = self.model(state)
        if item: output = Utils.argmax(output)
        return output

    def __compute_verbose_train(self, epoch, start_time, train_loss):
        print("Epoch [{}] took {:.2f}s | train_loss: {:.4f}".format(epoch, time.time() - start_time, train_loss))

    def __compute_early_stopping(self, epoch, my_es, train_loss):
        break_it = False
        my_es(train_loss)
        if my_es.early_stop:
            print(f'At epoch {epoch}, the second early stopping tolerance = {my_es.tolerance} has been reached,'
                  f' the training loss is not decreasing anymore -> stop it')
            break_it = True
        return break_it

    def fit(self, state: np.ndarray, target: torch.tensor):
        my_es = EarlyStopping()

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            train_loss = self.__train_model(state, target)

            break_it = self.__compute_early_stopping(epoch, my_es, train_loss)
            if break_it:
                break

            if not self.hyperopt:
                if self.verbose:
                    self.__compute_verbose_train(epoch, start_time, train_loss)
            else:
                pass
                # Send the current validation loss and accuration back to Tune for the hyperopt
                # Ray Tune can then use these metrics to decide which hyperparameter configuration lead to the best results.
                #tune.report(train_loss=train_loss_mean, train_acc=train_acc_mean, val_loss=val_loss_mean, val_acc=val_acc_mean)

            if self.save:
                torch.save(self.model.state_dict(), f'{self.path}/model_{epoch}.pt')

        if break_it:
            self.best_epoch = epoch - my_es.tolerance
        else:
            self.best_epoch = epoch
        self.train_loss = train_loss

    def predict(self, state: np.ndarray, item: bool=None):
        """
        if self.best_epoch is not None:
            epoch = self.best_epoch
        else:
            epoch = Utils.find_last_epoch(self.path)
        self.model = Utils.load_model(self.model, epoch, self.path, self.device)
        """
        prediction = self.__evaluate_model(state, item)
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
    def find_last_epoch(path):
        return int([f for f in os.listdir(path)][-1].split('_')[1].split('.')[0])

    @staticmethod
    def load_model(model, epoch, path, device):
        path_ = f'{path}/model_{epoch}.pt' if epoch is not None else path
        if device.type == 'cpu':
            model.load_state_dict(torch.load(path_, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(path_))
        return model
