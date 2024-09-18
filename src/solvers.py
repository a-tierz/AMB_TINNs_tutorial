import torch
import torch.nn as nn
import pytorch_lightning as pl


class Solver(pl.LightningModule):
    def __init__(self, base_model, criterion=nn.MSELoss(), dt=0.1, lr=1e-3, optimizer=torch.optim.Adam, check_val=1):
        super(Solver, self).__init__()

        self.model = base_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.miles = [100, 150]
        self.gamma = 1e-1
        self.check_val = check_val
        self.dt = torch.Tensor(dt)

    def forward(self, x):
        dzdt = self.model(x)
        return dzdt

    def training_step(self, batch, batch_idx):
        x, y = batch
        noise = 4e-5 * torch.randn_like(x)
        x = x + noise
        dzdt = self.forward(x)
        y_pred = x + self.dt.to(x.device) * dzdt
        loss = self.criterion(y_pred, y)
        self.log('train_loss', loss.detach().item(), on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        dzdt = self.forward(x)
        y_pred = x + self.dt.to(x.device) * dzdt
        loss = self.criterion(y_pred, y)
        self.log('val_loss', loss.detach().item(), on_epoch=True, on_step=False)

    def predict_step(self, x, g=None):
        dzdt = self.forward(x)
        y_pred = x + self.dt.to(x.device) * dzdt

        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                              milestones=self.miles,
                                                              gamma=self.gamma),
            'monitor': 'val_loss'}

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


class Solver_TINNS(pl.LightningModule):
    def __init__(self, base_model, dt=0.01, criterion=nn.MSELoss(), lr=1e-3, optimizer=torch.optim.Adam, check_val=1):
        super(Solver_TINNS, self).__init__()

        self.model = base_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.miles = [100, 150]
        self.gamma = 1e-1
        self.check_val = check_val
        self.dt = torch.Tensor(dt)
        self.lambda_d = 100

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        noise = 4e-5 * torch.randn_like(x)
        x = x + noise
        dzdt, deg_E, deg_S = self.forward(x)
        y_pred = x + self.dt.to(x.device) * dzdt

        loss_deg_E = (deg_E ** 2).mean()
        loss_deg_S = (deg_S ** 2).mean()
        loss_z = self.criterion(y_pred, y)
        loss = self.lambda_d * loss_z + (loss_deg_E + loss_deg_S)

        self.log('train_loss', loss.detach().item(), on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        dzdt, deg_E, deg_S = self.forward(x)
        y_pred = x + self.dt.to(x.device) * dzdt

        loss_deg_E = (deg_E ** 2).mean()
        loss_deg_S = (deg_S ** 2).mean()
        loss_z = self.criterion(y_pred, y)
        loss = self.lambda_d * loss_z + (loss_deg_E + loss_deg_S)

        self.log('val_loss', loss.detach().item(), on_epoch=True, on_step=False)

    def predict_step(self, x, g=None):
        dzdt, deg_E, deg_S = self.forward(x)
        y_pred = x + self.dt.to(x.device) * dzdt
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                              milestones=self.miles,
                                                              gamma=self.gamma),
            'monitor': 'val_loss'}

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
