import os

import numpy as np
import poutyne
import torch
import torch.nn.functional as F
from joblib import dump, load
from poutyne import Model as PoutyneModel, EarlyStopping, BestModelRestore
from torch.optim import AdamW
import matplotlib.pyplot as plt
from trainer.patches_funs import patchify, unpatchify, mask_patches # generate_random_patch_mask


def plot_fbank(fbank, title=None, cmap='inferno'):
    fig, axs = plt.subplots(1, 1)
    fig.set_figheight(4)
    fig.set_figwidth(8)
    axs.set_title(title or "Filter bank")
    cax1 = axs.imshow(fbank, aspect="auto", origin="lower", cmap=cmap) # inferno, hot
    axs.set_ylabel("Frequency")
    axs.set_xlabel("Time")
    fig.colorbar(cax1, ax=axs, orientation='vertical', label='Intensity')

def normalize_loss_map(mse_map, eps=1e-8):
    ''' using Min-Max Normalization '''

    norm_map = torch.zeros_like(mse_map)
    min_val = mse_map[0].min()
    max_val = mse_map[0].max()
    norm_map[0] = (mse_map[0] - min_val) / (max_val - min_val + eps)
    return norm_map

def apply_mask_and_perturb(x, mask, mask_value= 0.0):
    perturbed = x.clone()
    perturbed[mask] = mask_value
    return perturbed

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
    def __init__(self, num_input=1001, num_input_channels=64, start_num_channels=None, bottleneck_dim=None,
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
        self.start_num_channels = start_num_channels
        self.bottleneck_dim = bottleneck_dim
        self.maskedae = maskedae
        self.name = name
        self.num_per_hidden = num_per_hidden
        self.device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")
        if batch_size is not None:
            batch_size_to_use = batch_size
        self.batch_size = batch_size_to_use
        self.random_state = np.random.RandomState(42)
        self.model = None
        self.loss = loss
        self.lr = lr
        self.reduce_lr_patience = reduce_lr_patience
        self.stop_patience = stop_patience
        self.network_class = network_class
        self.verbose = verbose
        self.beta = beta
        self.latent_size = latent_size
        self.max_value = 9999.0
        self.min_value = 9999.0

    def load_model(self, name, default_model):
        model = default_model
        training_needed = True
        if os.path.exists(os.path.join(os.path.dirname(__file__), f'{name}.joblib')):
            model = load(os.path.join(os.path.dirname(__file__), f'{name}.joblib'))
            # print("model loaded...")
            training_needed = False
        return model, training_needed

    def save_model(self, name, model):
        dump(model, os.path.join(os.path.dirname(__file__), f'{name}.joblib'))

    def init_model(self):
        if self.bottleneck_dim is not None or self.start_num_channels is not None:
            network = self.network_class(self.input_size,
                                         input_num_chanels=self.num_input_channels,
                                         start_num_channels=self.start_num_channels, 
                                         bottleneck_dim=self.bottleneck_dim,
                                        )
        else:
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
            train_dataset_extracted = train_dataset[:]
            to_use_batch_size = min(len(val_dataset), self.batch_size)
            extracted_val = val_dataset[:]
            history = self.model.fit(train_dataset_extracted[0], train_dataset_extracted[1],
                           validation_data=(extracted_val[0], extracted_val[1]),
                           epochs=500, verbose=self.verbose, batch_size=to_use_batch_size,
                           callbacks=[
                               poutyne.CosineAnnealingWarmRestarts(T_0=5, eta_min=1e-5),
                               EarlyStopping(patience=self.stop_patience, min_delta=1e-5),
                               BestModelRestore(verbose=False)],
                           dataloader_kwargs={"shuffle": True})
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

    def get_loss(self, x, masked_x=None, ae_mask=None, #positions=None, patches=None, patchmasks=None, 
                 do_patch_mae=False, patch_size=(16, 16),
                 reduction='mean', title='fbank', plot=False, 
                 thresholding=False, percentile_value=0.7, 
                 interpret_attribution=False, attribution_mask=None):
        # x --> original_input

        if self.maskedae:
            if ae_mask is None or masked_x is None:
                raise ValueError("masked_x and ae_mask are required for masked prediction and masked loss respectively...")
            prediction = self.predict(masked_x)
            prediction = torch.tensor(prediction)
            mse = ((prediction - x) ** 2) * ae_mask
            reconstruction_error = mse.view(mse.size(0), -1).sum(dim=1) / ae_mask.view(ae_mask.size(0), -1).sum(dim=1) # reconstruction_error = <total masked MSE per sample> / <num masked elements per sample>
            
        if do_patch_mae:
            patches, positions = patchify(x[0], patch_size)
            N = len(patches)
            errors = []
        
            for i in range(N):
                # Mask only patch i
                mask = torch.zeros(N)
                mask[i] = 1
                masked_patches = mask_patches(patches, mask)
        
                # Reconstruct masked spectrogram
                masked_spec = unpatchify(masked_patches, positions, x[0].shape)
                # plot_fbank(masked_spec)
        
                # Model inference
                output = self.predict(torch.tensor(masked_spec, dtype=torch.float32).unsqueeze(0))
                output = torch.tensor(output)
        
                # Re-patchify reconstructed output
                rec_patches, _ = patchify(output[0], patch_size)
        
                # Compute error for the masked patch only
                # plot_fbank(patches[i])
                # plot_fbank(rec_patches[i])
                err = F.mse_loss(rec_patches[i], patches[i], reduction='mean')
                errors.append(err.item())
        
            # Reshape errors into patch grid
            H, W = x[0].shape
            ph, pw = patch_size
            num_rows = (H + ph - 1) // ph
            num_cols = (W + pw - 1) // pw
        
            error_map = torch.tensor(errors).view(num_rows, num_cols)
            plot_fbank(error_map, f'{title} error map', cmap='hot')
            return
        
        else:
            prediction = self.predict(x)
            prediction = torch.tensor(prediction)
            reconstruction_error = self.loss(prediction, x, reduction=reduction)

        if plot or thresholding:
            lossmap = self.loss(prediction, x, reduction='none')
        if plot:
            normalizedlossmap = normalize_loss_map(lossmap)
            # print('x shape', x.shape)
            # print('prediction shape', prediction.shape)
            # print('loss map shape', normalizedlossmap.shape)
            plot_fbank(x[0], f'{title} input')
            plot_fbank(prediction[0], f'{title} reconstructed output')
            plot_fbank(normalizedlossmap[0], f'normalized MSE loss map - {title}', cmap='hot')
            if self.maskedae:
                plot_fbank(masked_x[0], f'{title} masked input')
                plot_fbank(ae_mask[0], f'{title} mask')
        
        if thresholding:
            threshold = torch.quantile(lossmap, percentile_value)
            binary_mask = (lossmap > threshold).to(torch.uint8)
            plot_fbank(binary_mask[0], f"Error Regions (Binary Mask > {percentile_value*100}th Percentile)")

        if interpret_attribution:
            if masked_x is not None:
                # get prediction from masked input
                prediction = self.predict(masked_x)
                prediction = torch.tensor(prediction)

            # full reconstruction error
            reconstruction_error = self.loss(prediction, x, reduction=reduction)

            # MSE map
            mse_map = (prediction - x) ** 2
            
            # MSE inside and outside the mask
            mse_outside = mse_map[0][~attribution_mask].mean()
            mse_inside = mse_map[0][attribution_mask].mean()
            
            return reconstruction_error, mse_outside, mse_inside

        
        # if x.shape[0] == 1:
        #     return reconstruction_error.item()
        return reconstruction_error


