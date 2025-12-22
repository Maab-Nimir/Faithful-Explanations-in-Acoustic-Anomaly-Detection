import os
import matplotlib.pyplot as plt

import numpy as np
import poutyne
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from joblib import dump, load
from poutyne import Model as PoutyneModel, EarlyStopping, BestModelRestore, ModelCheckpoint

from torch.optim import AdamW

from poutyne import Callback

class LossPlotCallback(Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.train_loss = []
        self.val_loss = []
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Convert tensors to floats if needed
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        if torch.is_tensor(train_loss):
            train_loss = train_loss.item()
        if torch.is_tensor(val_loss):
            val_loss = val_loss.item()
        
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

        plt.figure()
        plt.plot(self.train_loss, label='Training Loss')
        plt.plot(self.val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training vs Validation Loss')
        plt.savefig(self.save_path, format='pdf', bbox_inches='tight')
        plt.close()



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
    def __init__(self, num_input=1001, num_input_channels=64, epochs=500, 
                 d_model=500, nhead=10, in_features=50,
                 maskedae=False, batch_size=16, num_per_hidden=256, #dcase_training=False,
                 lr=1e-7, latent_size=2,
                 reduce_lr_patience=15, stop_patience=30,
                 network_class=None, loss=F.mse_loss, verbose=True, beta=1.0, name="ae-1000", use_attention=True):
        super().__init__()
        batch_size_to_use = 256
        cuda_device = 0
        self.input_size = num_input
        self.use_attention = use_attention
        self.num_input_channels = num_input_channels
        self.nhead = nhead
        self.d_model = d_model
        self.in_features = in_features
        # self.dcase_training = dcase_training
        self.maskedae = maskedae
        self.name = name
        self.num_per_hidden = num_per_hidden
        self.device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")
        if batch_size is not None:
            batch_size_to_use = batch_size
        self.batch_size = batch_size_to_use
        self.model = None
        self.loss = loss
        self.lr = lr
        self.reduce_lr_patience = reduce_lr_patience
        self.stop_patience = stop_patience
        self.network_class = network_class
        self.verbose = verbose
        self.beta = beta
        self.latent_size = latent_size
        self.epochs = epochs

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
                                     d_model=self.d_model,
                                     nhead=self.nhead,
                                     in_features = self.in_features,
                                     # dcase_training = self.dcase_training,
                                    )
        opt = AdamW(network.parameters(), lr=self.lr)
        loss_fn = masked_mse_loss if self.maskedae else self.loss
        self.model = PoutyneModel(network, opt, loss_fn, device=self.device)

    def train(self, train_dataset, val_dataset):
        model, train_needed = self.load_model(f"{self.name}-{self.latent_size}", self.model)
        self.model = model
        if train_needed:
            # checkpoint path
            output_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
            os.makedirs(output_dir, exist_ok=True)
            checkpoint_path = os.path.join(output_dir, f"{self.name}_checkpoint.joblib")
            final_model_path = os.path.join(output_dir, f"{self.name}_final.joblib")
        
            # Initialize or resume model
            self.init_model()
        
            # If checkpoint exists, load it
            if os.path.exists(checkpoint_path):
                print(f"🔁 Resuming training from checkpoint: {checkpoint_path}")
                self.model.load_weights(checkpoint_path)
                # Resume from last completed epoch
                initial_epoch = 0
                if os.path.exists(checkpoint_path):
                    try:
                        checkpoint_data = load(checkpoint_path)  # Poutyne saves as a Joblib dict
                        if 'epoch' in checkpoint_data:
                            initial_epoch = checkpoint_data['epoch']
                            print(f"⏩ Resuming from epoch {initial_epoch + 1}")
                    except Exception as e:
                        print(f"⚠️ Could not read epoch from checkpoint: {e}")
            else:
                print("🚀 Starting new training run...")
                initial_epoch = 0

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

            # Define checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                filename=checkpoint_path,
                monitor='val_loss',     # metric to track
                mode='min',
                save_best_only=False,   
                period=2,               # save every 2 epochs
                verbose=True,
            )

            # Only create scheduler if training is starting from epoch zero
            scheduler_callback = None
            if initial_epoch == 0:
                scheduler_callback = poutyne.CosineAnnealingWarmRestarts(T_0=5, eta_min=1e-5)
                print("📈 Initialized new LR scheduler")
            else:
                print("⏸️ Using LR scheduler state restored from checkpoint")
            
            # Build callback list dynamically
            callbacks = [
                EarlyStopping(patience=self.stop_patience, min_delta=1e-5),
                BestModelRestore(verbose=False),
                checkpoint_callback,
            ]
            if scheduler_callback:
                callbacks.insert(0, scheduler_callback)  # only add new one if needed
                
            plot_callback = LossPlotCallback(os.path.join(os.path.dirname(__file__), f"track_{self.name}_loss_curve.pdf"))
            callbacks.append(plot_callback)
            
            history = self.model.fit_generator(train_loader,
                                               val_loader,
                                               epochs=self.epochs, initial_epoch=initial_epoch, verbose=self.verbose,
                                               callbacks=callbacks)
            
            # save the final model after training stops due to early stopping or reaching the number of epochs
            self.model.save_weights(final_model_path)
            print(f"✅ Final model saved to: {final_model_path}")
            
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
            # plt.show()
            # Define output path
            output_dir = os.path.dirname(__file__) # the directory where the current Python script file is located
            pdf_path = os.path.join(output_dir, f"{self.name}_loss_curve.pdf")
            
            # Save to PDF
            plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            
            # Save the trained model
            self.save_model(f"{self.name}-{self.latent_size}", self.model)

    def predict(self, x):
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


