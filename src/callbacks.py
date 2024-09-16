import torch
import numpy as np
import pytorch_lightning as pl
from src.utils import plot_system
import matplotlib.pyplot as plt

# Add this callback to control plotting every X fraction of epochs
class PlotEveryNEpochs(pl.Callback):
    def __init__(self, y_data, n_epochs_fraction=4):
        super().__init__()
        self.y_data = y_data
        self.n_epochs_fraction = n_epochs_fraction

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        total_epochs = trainer.max_epochs

        # Plot if the current epoch is a multiple of 1/n_epochs_fraction
        if current_epoch % (total_epochs // self.n_epochs_fraction) == 0:
            self.plot_prediction(trainer, pl_module)

    def plot_prediction(self, trainer, pl_module):
        with torch.no_grad():
            y_pred = []
            pred_u = model_dd(torch.tensor(Z[:, :-1].T, dtype=torch.float32)[0, :])
            y_pred.append(pred_u.ravel().detach().numpy())
            for i in range(Z[:, :-1].shape[1]):
                pred_u = model_dd(pred_u)
                y_pred.append(pred_u.ravel().detach().numpy())

            # Concatenate predictions over all batches
            y_pred = np.array(y_pred).T
            plot_system(self.y_data, y_pred)


        # Return model back to training mode
        pl_module.train()



class LossHistoryCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Almacena la pérdida de entrenamiento del último epoch
        avg_train_loss = trainer.callback_metrics["train_loss"].item()
        self.train_losses.append(avg_train_loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Almacena la pérdida de validación del último epoch
        avg_val_loss = trainer.callback_metrics["val_loss"].item()
        for i in range(trainer.lightning_module.check_val):
            self.val_losses.append(avg_val_loss)

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.yscale('log')
        plt.legend()
        plt.savefig('plots/training_BlackBox.png')
        plt.show()
