import os

import numpy as np
import poutyne
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from joblib import dump, load
from poutyne import Model as PoutyneModel, EarlyStopping, BestModelRestore

from torch.optim import AdamW
# import matplotlib.pyplot as plt
# from trainer.patches_funs import patchify, unpatchify, mask_patches # generate_random_patch_mask

# from scipy.ndimage import gaussian_filter1d

# def plot_cosine_distance_over_time(distance, title="Reconstruction Cosine Distance Over Time"):
#     """
#     Plot cosine distance over time with labels and title.

#     Args:
#         distance: Tensor or numpy array of shape [time]
#         title: Plot title
#     """
#     time_steps = range(distance.shape[0])
#     plt.figure(figsize=(10, 4))
#     plt.plot(time_steps, distance.cpu().numpy(), color='tab:blue')
#     plt.xlabel("Time")
#     plt.ylabel("Cosine Distance (1 - similarity)")
#     plt.title(title)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# def normalize_loss_map(mse_map, eps=1e-8):
#     ''' using Min-Max Normalization '''

#     norm_map = torch.zeros_like(mse_map)
#     min_val = mse_map[0].min()
#     max_val = mse_map[0].max()
#     norm_map[0] = (mse_map[0] - min_val) / (max_val - min_val + eps)
#     return norm_map

# def apply_mask_and_perturb(x, mask, mask_value= 0.0):
#     perturbed = x.clone()
#     perturbed[mask] = mask_value
#     return perturbed

def masked_mse_loss(prediction, original_input_and_mask):
    original_input, mask = original_input_and_mask
    mse = (prediction - original_input) ** 2
    return (mse * mask).sum() / mask.sum()

# def compute_error_curve(binary_mask, axis=1):
#     """Sum binary mask along given axis (1: frequency → error over time, 2: time → error over frequency)."""
#     return binary_mask.sum(dim=axis)[0].float()

# def plot_error_with_peaks(curve, norm_curve, peaks, threshold, axis_label='Time', title='Anomaly Activity Curve'):
#     """Plot normalized error curve with detected peaks."""
#     plt.figure(figsize=(12, 5))
#     # plt.plot(curve.cpu().numpy(), label=f'Error over {axis_label}', color='green', alpha=0.6)
#     plt.plot(norm_curve, label=f'Normalized Error over {axis_label}', color='blue')
#     plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.2f})')
#     plt.scatter(peaks, norm_curve[peaks], color='red', s=80, label='Anomaly Peaks', zorder=5)
#     plt.title(title)
#     plt.xlabel(axis_label)
#     plt.ylabel('Anomaly Score')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# def process_error_analysis(binary_mask, axis, peaks_percentile, peaks_distance, axis_label, plot=False):
#     """Wrapper to compute, normalize, and plot error over a given axis."""
#     curve = compute_error_curve(binary_mask, axis=axis)
#     norm_curve, peaks, threshold = normalize_and_find_peaks(curve, percentile=peaks_percentile, distance=peaks_distance)
#     if plot:
#         plot_error_with_peaks(curve, norm_curve, peaks, threshold, axis_label=axis_label)

# def inject_frequency_anomaly(spec, artificial_signal_params=None):
#     """
#     Injects a horizontal line (constant frequency anomaly) across time.
    
#     Parameters:
#     - spec: torch.Tensor of shape (F, T)
#     - artificial_signal_params:-
#         - freq_idx: int, frequency bin to inject
#         - start_time, end_time: int, time range
#         - intensity: float, how strong the anomaly is
    
#     Returns:
#     - Modified spectrogram
#     """
#     # freq_idx, start_time, end_time, intensity= artificial_signal_params
#     spec_with_anomaly = spec.clone()
#     for freq_idx, start_time, end_time, intensity in artificial_signal_params:
#         spec_with_anomaly[freq_idx, start_time:end_time] += intensity
#     return spec_with_anomaly

# def ablate_encoder_block(model, block_index):
#     """
#     Returns a list of hooks that zero out the output of the encoder block at block_index.
#     Assumes model has .encoder_blocks
#     """
#     hooks = []

#     def hook_fn(module, input, output):
#         return torch.zeros_like(output)

#     if hasattr(model, 'encoder_blocks') and block_index < len(model.encoder_blocks):
#         h = model.encoder_blocks[block_index].register_forward_hook(hook_fn)
#         hooks.append(h)
#     else:
#         raise ValueError(f"Encoder block {block_index} not found in model.")
#     return hooks
# def plot_layerwise_attribution(attribution_scores, baseline_score):
#     plt.figure(figsize=(10, 5))
#     plt.bar(range(len(attribution_scores)), attribution_scores, color='orange')
#     plt.axhline(y=0, color='black', linestyle='--')
#     plt.ylabel("# Peaks in Human Annotations")
#     plt.xlabel("Ablated Encoder Layer Index")
#     plt.title(f"Layer Attribution via Peak Drop (Baseline = {baseline_score} Peaks)")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

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

    def _layerwise_ablation(self, x, annotations, percentile_value=0.95, peaks_percentile=98, peaks_distance=3):
        """
        Performs ablation on encoder layers and returns peak counts inside annotated regions.
    
        Returns:
            attribution_scores (List[int]): Number of anomaly peaks inside annotations after ablation
        """
        # Baseline
        pred = torch.tensor(self.predict(x))
        lossmap = self.loss(pred, x, reduction='none')
        threshold = torch.quantile(lossmap, percentile_value)
        binary_mask = (lossmap > threshold).to(torch.uint8)
    
        # Error curve and peak detection
        curve = compute_error_curve(binary_mask, axis=1)
        norm_curve, peaks, _ = normalize_and_find_peaks(curve, percentile=peaks_percentile, distance=peaks_distance)
        baseline_count = sum(any(start <= p < end for (start, end) in annotations) for p in peaks)
        # baseline_peak_count = sum([(start <= p < end) for p in peaks for (start, end) in annotations]) ???
        # baseline_peak_score = baseline_peak_count/max(len(peaks), 1)
        # print('baseline_peak_score = ', baseline_peak_score, ' peaks= ', len(peaks), ' baseline_peak_count  inside annotated mask= ', baseline_peak_count)
    
        attribution_scores = []
        layers_to_ablate = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'conv9', 'conv10', 'transformer_encode']
        original_state = self.model.network.state_dict()
    
        for layer_name in layers_to_ablate:
            self.model.network.load_state_dict(original_state)
            layer = getattr(self.model.network, layer_name)
    
            def ablation_hook(module, input, output):
                return torch.zeros_like(output)
    
            hook = layer.register_forward_hook(ablation_hook)
    
            # Predict with ablated layer
            pred_ablate = torch.tensor(self.predict(x))
            lossmap_ablate = self.loss(pred_ablate, x, reduction='none')
            threshold = torch.quantile(lossmap_ablate, percentile_value) # i thin the threshold should be dependent on the corresponding error map
            binary_mask_ablate = (lossmap_ablate > threshold).to(torch.uint8)
            curve_ablate = compute_error_curve(binary_mask_ablate, axis=1)
            norm_curve_ablate, peaks_ablate, _ = normalize_and_find_peaks(curve_ablate, percentile=peaks_percentile, distance=peaks_distance)
    
            count = sum(any(start <= p < end for (start, end) in annotations) for p in peaks_ablate)
            # count = sum([(start <= p < end) for p in peaks for (start, end) in annotations]) ??? which one is correct or are they similar?
            attribution_scores.append(count)
    
            print(f"[Ablation] {layer_name}: {count} peaks in annotated regions")
            hook.remove()
    
        plot_layerwise_attribution(attribution_scores, baseline_count)
        return attribution_scores

    def get_loss(self, x, masked_x=None, ae_mask=None, #positions=None, patches=None, patchmasks=None, 
                 do_patch_mae=False, patch_size=(16, 16),
                 reduction='mean', title='fbank',
                 threshold_2d_error_map=False, percentile_value=0.95, peaks_percentile=98, peaks_distance=3,
                 attribution_mask=None, annotations=None,
                 artificial_signal_params=None,
                 layer_attribution=False, return_2d_error_map=False,
                ):
        # x --> original_input

        if layer_attribution:
            if annotations is None:
                raise ValueError("Annotations required for layerwise attribution.")
            attribution_scores = self._layerwise_ablation(x, annotations, percentile_value, peaks_percentile, peaks_distance)
            return attribution_scores

        if self.maskedae:
            if ae_mask is None or masked_x is None:
                raise ValueError("masked_x and ae_mask are required for masked prediction and masked loss respectively...")
            prediction = self.predict(masked_x)
            prediction = torch.tensor(prediction)
            mse = ((prediction - x) ** 2) * ae_mask
            print(f"reconstruction_error inside the mask, self.maskedae = {self.maskedae}")
            reconstruction_error = mse.view(mse.size(0), -1).sum(dim=1) / ae_mask.view(ae_mask.size(0), -1).sum(dim=1) # reconstruction_error = <total masked MSE per sample> / <num masked elements per sample>
            
        elif do_patch_mae:
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
            print(f"do_patch_mae = {do_patch_mae}")
            return

        elif artificial_signal_params is not None:
            x = inject_frequency_anomaly(x[0], artificial_signal_params).unsqueeze(0)
            prediction = self.predict(x)
            prediction = torch.tensor(prediction)
            reconstruction_error = self.loss(prediction, x, reduction=reduction)
            print(f"artificial signal full reconstruction_error")
        
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


