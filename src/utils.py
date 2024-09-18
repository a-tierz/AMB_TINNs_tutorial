
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import torch


# Define colors for the plots
color_x_gt = 'deepskyblue'
color_y_gt = 'sandybrown'
color_x_net = 'skyblue'
color_y_net = 'darkorange'


def plot_system(y_data, y_net=None, name='.png', error=0):
    "Pretty plot training results"
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # First subplot: Pendulum position (x, y)
    axs[0].plot(y_data[0,:], label="Ground truth x", color=color_x_gt)
    axs[0].plot(y_data[1,:], label="Ground truth y", color=color_y_gt)
    if y_net is not None:
      axs[0].plot(y_net[0,:], label="Net prediction x", color=color_x_net, linestyle='--')
      axs[0].plot(y_net[1,:], label="Net prediction y", color=color_y_net, linestyle='--')

    axs[0].set_title("Pendulum positions")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Positon [-]")
    # axs[0].set_ylim(-6.5, 1.5)
    axs[0].legend()

    # Second subplot: Pendulum velocity (vx, vy)
    axs[1].plot(y_data[2,:], label="Ground truth vx", color=color_x_gt)
    axs[1].plot(y_data[3,:], label="Ground truth vy", color=color_y_gt)
    if y_net is not None:
      axs[1].plot(y_net[2,:], label="Net prediction vx", color=color_x_net, linestyle='--')
      axs[1].plot(y_net[3,:], label="Net prediction vy", color=color_y_net, linestyle='--')

    axs[1].set_title("Pendulum velocity")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Velocity [-]")
    # axs[1].set_ylim(-8.0,8.0)
    axs[1].legend()

    # Add a large title to the figure
    fig.suptitle(f"MAError: {error}", fontsize=16)

    # show plots
    plt.tight_layout()
    plt.savefig(name)
    plt.show()


def generateDatasets(X, Y, batch_size):
  # split data into train and validatio
  train_t, valid_t, train_u, valid_u = train_test_split(X.T, Y.T, test_size=0.15, random_state=42)

  # Convert data to PyTorch tensors
  train_t = torch.tensor(train_t, dtype=torch.float32)
  train_u = torch.tensor(train_u, dtype=torch.float32)
  valid_t = torch.tensor(valid_t, dtype=torch.float32)
  valid_u = torch.tensor(valid_u, dtype=torch.float32)

  # Create TensorDatasets
  train_dataset = TensorDataset(train_t, train_u)
  valid_dataset = TensorDataset(valid_t, valid_u)

  # Create DataLoaders
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

  return train_loader, valid_loader

