import os
import matplotlib.pyplot as plt
import numpy as np
import poutyne
import torch
import torch.nn.functional as F
from joblib import dump, load
from poutyne import Model as PoutyneModel, EarlyStopping, BestModelRestore, ModelCheckpoint
from torch.optim import AdamW

class Model:
    def train(self, x, y):
        pass

    def predict(self, x):
        pass

    def score(self, x, y):
        pass


class ModelTrainer(Model):
    def __init__(self, num_input=1001, num_input_channels=64, d_model=500, nhead=10, in_features=50,
                 batch_size=16, num_per_hidden=256,
                 lr=1e-5, latent_size=2,
                 reduce_lr_patience=15, stop_patience=30,
                 network_class=None, loss=F.mse_loss, verbose=True, beta=1.0, name="ae-1000", use_attention=True):
        super().__init__()
        batch_size_to_use = 256
        cuda_device = 0
        self.input_size = num_input
        self.use_attention = use_attention
        self.num_input_channels = num_input_channels
        self.d_model = d_model
        self.nhead = nhead
        self.in_features = in_features
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
            training_needed = False
        return model, training_needed

    def save_model(self, name, model):
        dump(model, os.path.join(os.path.dirname(__file__), f'{name}.joblib'))

    def init_model(self):
        network = self.network_class(self.input_size,
                                     input_num_chanels=self.num_input_channels,
                                     d_model=self.d_model,
                                     nhead = self.nhead,
                                     in_features = self.in_features,
                                    )
        opt = AdamW(network.parameters(), lr=self.lr)
        self.model = PoutyneModel(network, opt, self.loss, device=self.device)

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
            
            # self.init_model()
            train_dataset_extracted = train_dataset[:]
            to_use_batch_size = min(len(val_dataset), self.batch_size)
            extracted_val = val_dataset[:]
            
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
            
            history = self.model.fit(train_dataset_extracted, train_dataset_extracted,
                           validation_data=(extracted_val, extracted_val),
                           epochs=500, initial_epoch=initial_epoch,
                           verbose=self.verbose, batch_size=to_use_batch_size,
                           callbacks=callbacks,
                           dataloader_kwargs={"shuffle": True})
            
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
            # Define output path
            output_dir = os.path.dirname(__file__)
            # output_dir = os.path.join(os.path.dirname(__file__), "results")
            os.makedirs(output_dir, exist_ok=True)
            pdf_path = os.path.join(output_dir, f"{self.name}_loss_curve.pdf")
            plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
            plt.close()
            
            # Save the trained model
            self.save_model(f"{self.name}-{self.latent_size}", self.model)

    def predict(self, x):
        predictions = self.model.predict(x)
        return predictions

    def get_loss(self, x, reduction='mean'):
        prediction = self.predict(x)
        prediction = torch.tensor(prediction)
        reconstruction_error = self.loss(prediction, x, reduction=reduction)
        # if x.shape[0] == 1:
        #     return reconstruction_error.item()
        return reconstruction_error

