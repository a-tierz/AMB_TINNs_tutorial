import scipy.io as sio

import numpy as np
import torch

import pytorch_lightning as pl

from src.models import BlackBox, TINN_01, TINN_02, TINN_03, TINN, Solver, Solver_TINNS
from src.callbacks import LossHistoryCallback, PlotEveryNEpochs
from src.utils import generateDatasets, plot_system

# ['BlackBox', 'TINN_01', 'TINN_02', 'TINN_03', 'TINN']
model_name = 'BlackBox'
# ['no_dissipation', 'dissipation', 'double_pen']
dataset_type = 'double_pen'

# Load the file using sio.loadmat
if dataset_type == 'dissipation':
    data = sio.loadmat('data/db_pendulum_dis_train.mat')
    data_test = sio.loadmat('data/db_pendulum_dis_test.mat')
elif dataset_type == 'no_dissipation':
    data = sio.loadmat('data/db_pendulum_nodis_train.mat')
    data_test = sio.loadmat('data/db_pendulum_nodis_test.mat')
elif dataset_type == 'double_pen':
    data = sio.loadmat('data/database_double_pendulum_train.mat')
    data_test = sio.loadmat('data/database_double_pendulum_test.mat')

epochs = 200
lr = 1e-3
optm = torch.optim.Adam
batch_size = 250
check_val = 1

dInfo = {
    'dim_input': 10,  # Input size (e.g., size of the feature vector)  10
    'dim_hidden': 64,  # Size of the hidden layers
    'num_layers': 4  # Number of hidden layers
}

device = 'gpu' if torch.cuda.is_available() else 'cpu'

# Access the variables
Z = data['X']
Y = data['Y']
dt = data['dt']  # time increment
Z_test = data_test['X']
Y_test = data_test['Y']
n_sim = 199

train_loader, valid_loader = generateDatasets(Z, Y, batch_size)

# Add the loss callback to the trainer
loss_callback = LossHistoryCallback()

# Add the callback to the trainer to plot every 1/4 of the epochs
plot_callback = PlotEveryNEpochs(y_data=Z, n_epochs_fraction=4)

# Añadir el callback en tu trainer
loss_callback = LossHistoryCallback()

# Regularization techniques
early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=100, mode='min')

if model_name == 'BlackBox':
    blackbox_model = BlackBox(dInfo)
elif model_name == 'TINN_01':
    blackbox_model = TINN_01(dInfo)
elif model_name == 'TINN_02':
    blackbox_model = TINN_02(dInfo)
elif model_name == 'TINN_03':
    blackbox_model = TINN_03(dInfo)
elif model_name == 'TINN':
    blackbox_model = TINN(dInfo)

if model_name == 'TINN_03' or model_name == 'TINN':
    model_dd = Solver_TINNS(blackbox_model, dt=dt, lr=lr, optimizer=optm, check_val=check_val)
else:
    model_dd = Solver(blackbox_model, dt=dt, lr=lr, optimizer=optm, check_val=check_val)

# Para cargar el modelo sin haber entrenado
# model_dd.load_state_dict(torch.load(f'./model_{model_name}.pth', weights_only=True))

# Trainer setup
trainer_dd = pl.Trainer(
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    max_epochs=epochs,
    callbacks=[loss_callback, early_stopping],  # plot_callback
    check_val_every_n_epoch=check_val)

# Training the model
trainer_dd.fit(model_dd, train_loader, valid_loader)

loss_callback.plot_losses()

# Save the model
torch.save(model_dd.state_dict(), f'./model_{model_name}.pth')

# set model to evaluate
model_dd.eval()

# Make the rollback prediction
results = []
pred_u = model_dd.predict_step(torch.tensor(Z.T, dtype=torch.float32)[0, :].unsqueeze(0))
results.append(pred_u.ravel().detach().numpy())
for i in range(n_sim):
    pred_u = model_dd.predict_step(pred_u)
    results.append(pred_u.ravel().detach().numpy())

Z_pred = np.asarray(results).T
error = np.mean(np.abs(Z[:, :n_sim] - Z_pred[:, :n_sim]))
plot_system(Y[:, :n_sim], Z_pred, name=f'plots/{dataset_type}/train_{model_name}.png', error=error)

# Make the rollback prediction
results = []
pred_u = model_dd.predict_step(torch.tensor(Z_test.T, dtype=torch.float32)[0, :].unsqueeze(0))
results.append(pred_u.ravel().detach().numpy())
for i in range(n_sim):
    pred_u = model_dd.predict_step(pred_u)
    results.append(pred_u.ravel().detach().numpy())

Z_pred = np.asarray(results).T
error = np.mean(np.abs(Z_test[:, :n_sim] - Z_pred[:, :n_sim]))

plot_system(Y_test[:, :n_sim], Z_pred, name=f'plots/{dataset_type}/test_{model_name}.png', error=error)
