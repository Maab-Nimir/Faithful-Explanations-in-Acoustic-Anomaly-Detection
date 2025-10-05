import os

import numpy as np
import poutyne
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from joblib import dump, load
from poutyne import Model as PoutyneModel, EarlyStopping, BestModelRestore

from torch.optim import AdamW

def masked_mse_loss(prediction, original_input_and_mask):
    original_input, mask = original_input_and_mask
    mse = (prediction - original_input) ** 2
    return (mse * mask).sum() / mask.sum()

class Model:
    def train(self, x, y):
        pass

    def predict(self, x):
        pass

    def score(self, x, y):
        pass


class ModelTrainer(Model):
    def __init__(self, num_input=1001, num_input_channels=64, #start_num_channels=None, bottleneck_dim=None,
                 maskedae=False, batch_size=16, num_per_hidden=256,
                 lr=1e-5, latent_size=2,
                 reduce_lr_patience=15, stop_patience=30,
                 network_class=None, loss=F.mse_loss, verbose=True, beta=1.0, name="ae-1000", use_attention=True):
        super().__init__()
        batch_size_to_use = 256
        cuda_device = 0
        self.input_size = num_input
        self.use_attention = use_attention
        self.num_input_channels = num_input_channels
        self.maskedae = maskedae
        self.name = name
        self.num_per_hidden = num_per_hidden
        self.device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")
        if batch_size is not None:
            batch_size_to_use = batch_size
        self.batch_size = batch_size_to_use
        # self.random_state = np.random.RandomState(42)
        self.model = None
        self.loss = loss
        self.lr = lr
        self.reduce_lr_patience = reduce_lr_patience
        self.stop_patience = stop_patience
        self.network_class = network_class
        self.verbose = verbose
        self.beta = beta
        self.latent_size = latent_size
        # self.max_value = 9999.0
        # self.min_value = 9999.0

    def load_model(self, name, default_model):
        model = default_model
        training_needed = True
        if os.path.exists(os.path.join(os.path.dirname(__file__), f'{name}.joblib')):
            model = load(os.path.join(os.path.dirname(__file__), f'{name}.joblib'))
            training_needed = False
        return model, training_needed

    def save_model(self, name, model):
        dump(model, os.path.join(os.path.dirname(__file__), f'{name}.joblib'))

    def init_model(self):
        network = self.network_class(self.input_size,
                                 input_num_chanels=self.num_input_channels,
                                    )
        opt = AdamW(network.parameters(), lr=self.lr)
        loss_fn = masked_mse_loss if self.maskedae else self.loss
        self.model = PoutyneModel(network, opt, loss_fn, device=self.device)

    def train(self, train_dataset, val_dataset):
        model, train_needed = self.load_model(f"{self.name}-{self.latent_size}", self.model)
        self.model = model
        if train_needed:
            self.init_model()
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
            
            history = self.model.fit_generator(train_loader,
                                               val_loader,
                                               epochs=500, verbose=self.verbose,
                                               callbacks=[
                                                   poutyne.CosineAnnealingWarmRestarts(T_0=5, eta_min=1e-5),
                                                   EarlyStopping(patience=self.stop_patience, min_delta=1e-5),
                                                   BestModelRestore(verbose=False)])
            
            # Plot training history
            plt.figure()
            train_loss = [epoch_metrics['loss'] for epoch_metrics in history]
            val_loss = [epoch_metrics['val_loss'] for epoch_metrics in history]
            plt.plot(train_loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training vs Validation Loss')
            plt.show()
            self.save_model(f"{self.name}-{self.latent_size}", self.model)

    def predict(self, x):
        # print('predict fun ***********')
        # print(self.model.predict(x).shape)
        predictions = self.model.predict(x)
        return predictions

    def get_loss(self, x, masked_x=None, ae_mask=None, 
                 reduction='mean', return_2d_error_map=False,
                ):
        # x --> original_input
        if self.maskedae:
            if ae_mask is None or masked_x is None:
                raise ValueError("masked_x and ae_mask (the input mask) are required for masked prediction and masked loss respectively...")
            prediction = self.predict(masked_x)
            prediction = torch.tensor(prediction)
            mse = ((prediction - x) ** 2) * ae_mask
            print(f"reconstruction_error inside the mask, self.maskedae = {self.maskedae}")
            reconstruction_error = mse.view(mse.size(0), -1).sum(dim=1) / ae_mask.view(ae_mask.size(0), -1).sum(dim=1) # reconstruction_error = <total masked MSE per sample> / <num masked elements per sample>
        
        else:
            prediction = self.predict(x)
            prediction = torch.tensor(prediction)
            reconstruction_error = self.loss(prediction, x, reduction=reduction)
            # print(f"full reconstruction_error ")

        lossmap = self.loss(prediction, x, reduction='none')
        if return_2d_error_map:
            return lossmap
        
        # if x.shape[0] == 1:
        #     return reconstruction_error.item()
        return reconstruction_error


