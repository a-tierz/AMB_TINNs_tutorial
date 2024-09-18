import torch
import numpy as np
import scipy.io as sio
import pytorch_lightning as pl
import argparse

from src.models import BlackBox, TINN_01, TINN_02, TINN_03, TINN
from src.solvers import Solver, Solver_TINNS
from src.callbacks import LossHistoryCallback, PlotEveryNEpochs
from src.utils import generateDatasets, plot_system


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--model_name', type=str, default='BlackBox',
                        choices=['BlackBox', 'TINN_01', 'TINN_02', 'TINN_03', 'TINN'],
                        help='Select the model to use for training.')
    parser.add_argument('--dataset_type', type=str, default='no_dissipation',
                        choices=['no_dissipation', 'dissipation', 'double_pen'],
                        help='Select the type of dataset.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train the model.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=250, help='Batch size for training.')
    parser.add_argument('--check_val', type=int, default=1, help='Frequency of validation checks.')
    return parser.parse_args()


def main():
    args = parse_args()

    dInfo = {'dim_input': 4, 'dim_hidden': 64, 'num_layers': 4}
    n_sim = 199
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    optm = torch.optim.Adam

    # Datasets mapping
    dataset_paths = {
        'dissipation': ('data/db_pendulum_dis_train.mat', 'data/db_pendulum_dis_test.mat'),
        'no_dissipation': ('data/db_pendulum_nodis_train.mat', 'data/db_pendulum_nodis_test.mat'),
        'double_pen': ('data/database_double_pendulum_train.mat', 'data/database_double_pendulum_test.mat')
    }

    # Load datasets
    train_data_path, test_data_path = dataset_paths[args.dataset_type]
    data = sio.loadmat(train_data_path)
    data_test = sio.loadmat(test_data_path)

    # Extract data
    Z, Y, dt = data['X'], data['Y'], data['dt']
    Z_test, Y_test = data_test['X'], data_test['Y']

    # Generate loaders
    train_loader, valid_loader = generateDatasets(Z, Y, args.batch_size)

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
    blackbox_model = model_mapping[args.model_name](dInfo)

    # Select Solver
    if args.model_name in ['TINN_03', 'TINN']:
        model_dd = Solver_TINNS(blackbox_model, dt=dt, lr=args.lr, optimizer=optm, check_val=args.check_val)
    else:
        model_dd = Solver(blackbox_model, dt=dt, lr=args.lr, optimizer=optm, check_val=args.check_val)

    # Trainer setup
    trainer_dd = pl.Trainer(
        accelerator=device,
        max_epochs=args.epochs,
        callbacks=[loss_callback, early_stopping],  # Optional: add plot_callback
        check_val_every_n_epoch=args.check_val
    )

    # Train the model
    trainer_dd.fit(model_dd, train_loader, valid_loader)
    loss_callback.plot_losses()

    # Save the trained model
    torch.save(model_dd.state_dict(), f'data/weights/model_{args.model_name}_{args.dataset_type}.pth')

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
    plot_system(Y[:, :n_sim], Z_pred_train, name=f'plots/{args.dataset_type}/train_{args.model_name}.png',
                error=error_train)

    # Perform rollback prediction on test data
    Z_pred_test = rollback_prediction(model_dd, Z_test, n_sim)
    error_test = np.mean(np.abs(Z_test[:, :n_sim] - Z_pred_test[:, :n_sim]))
    plot_system(Y_test[:, :n_sim], Z_pred_test, name=f'plots/{args.dataset_type}/test_{args.model_name}.png',
                error=error_test)


if __name__ == "__main__":
    main()
