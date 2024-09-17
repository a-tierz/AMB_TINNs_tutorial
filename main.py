import torch
import numpy as np
import scipy.io as sio
import pytorch_lightning as pl

from src.models import BlackBox, TINN_01, TINN_02, TINN_03, TINN
from src.solvers import Solver, Solver_TINNS
from src.callbacks import LossHistoryCallback, PlotEveryNEpochs
from src.utils import generateDatasets, plot_system

# Configuration
model_name = 'TINN'  # ['BlackBox', 'TINN_01', 'TINN_02', 'TINN_03', 'TINN']
dataset_type = 'double_pen'  # ['no_dissipation', 'dissipation', 'double_pen']
epochs = 200
lr = 1e-3
batch_size = 250
check_val = 1
dInfo = {'dim_input': 10, 'dim_hidden': 64, 'num_layers': 4}
device = 'gpu' if torch.cuda.is_available() else 'cpu'
optm = torch.optim.Adam
n_sim = 199

# Datasets mapping
dataset_paths = {
    'dissipation': ('data/db_pendulum_dis_train.mat', 'data/db_pendulum_dis_test.mat'),
    'no_dissipation': ('data/db_pendulum_nodis_train.mat', 'data/db_pendulum_nodis_test.mat'),
    'double_pen': ('data/database_double_pendulum_train.mat', 'data/database_double_pendulum_test.mat')
}

# Load datasets
train_data_path, test_data_path = dataset_paths[dataset_type]
data = sio.loadmat(train_data_path)
data_test = sio.loadmat(test_data_path)

# Extract data
Z, Y, dt = data['X'], data['Y'], data['dt']
Z_test, Y_test = data_test['X'], data_test['Y']

# Generate loaders
train_loader, valid_loader = generateDatasets(Z, Y, batch_size)

# Callbacks
loss_callback = LossHistoryCallback()
plot_callback = PlotEveryNEpochs(y_data=Z, n_epochs_fraction=4)
early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=100, mode='min')

# Model mapping
model_mapping = {
    'BlackBox': BlackBox,
    'TINN_01': TINN_01,
    'TINN_02': TINN_02,
    'TINN_03': TINN_03,
    'TINN': TINN
}

# Instantiate model
blackbox_model = model_mapping[model_name](dInfo)

# Select Solver
if model_name in ['TINN_03', 'TINN']:
    model_dd = Solver_TINNS(blackbox_model, dt=dt, lr=lr, optimizer=optm, check_val=check_val)
else:
    model_dd = Solver(blackbox_model, dt=dt, lr=lr, optimizer=optm, check_val=check_val)

# Trainer setup
trainer_dd = pl.Trainer(
    accelerator=device,
    max_epochs=epochs,
    callbacks=[loss_callback, early_stopping],  # Optional: add plot_callback
    check_val_every_n_epoch=check_val
)

# Train the model
trainer_dd.fit(model_dd, train_loader, valid_loader)
loss_callback.plot_losses()

# Save the trained model
torch.save(model_dd.state_dict(), f'data/weights/model_{model_name}_{dataset_type}.pth')

# Model evaluation
model_dd.eval()


# Prediction function
def rollback_prediction(model, data, n_sim):
    results = []
    pred_u = model.predict_step(torch.tensor(data.T, dtype=torch.float32)[0, :].unsqueeze(0))
    results.append(pred_u.ravel().detach().numpy())

    for _ in range(n_sim):
        pred_u = model.predict_step(pred_u)
        results.append(pred_u.ravel().detach().numpy())

    return np.asarray(results).T


# Perform rollback prediction on train data
Z_pred_train = rollback_prediction(model_dd, Z, n_sim)
error_train = np.mean(np.abs(Z[:, :n_sim] - Z_pred_train[:, :n_sim]))
plot_system(Y[:, :n_sim], Z_pred_train, name=f'plots/{dataset_type}/train_{model_name}.png', error=error_train)

# Perform rollback prediction on test data
Z_pred_test = rollback_prediction(model_dd, Z_test, n_sim)
error_test = np.mean(np.abs(Z_test[:, :n_sim] - Z_pred_test[:, :n_sim]))
plot_system(Y_test[:, :n_sim], Z_pred_test, name=f'plots/{dataset_type}/test_{model_name}.png', error=error_test)
