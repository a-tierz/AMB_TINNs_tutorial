import torch
import torch.nn as nn
import numpy as np
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


class MLP(nn.Module):
    def __init__(self, layer_vec):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for k in range(len(layer_vec) - 1):
            layer = nn.Linear(layer_vec[k], layer_vec[k + 1])
            self.layers.append(layer)
            # if k != len(layer_vec) - 2: self.layers.append(nn.ReLU())
            # if k != len(layer_vec) - 2: self.layers.append(nn.LeakyReLU())
            if k != len(layer_vec) - 2: self.layers.append(nn.SiLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


'''
We define the model with an input layer, a number (defined) of intermediate
hidden layers, and an output layer. The output layer has de same dimmension than
the dimmensions of our problem (input dimmensions)
'''


class BlackBox(nn.Module):
    "Defines a MLP network"

    def __init__(self, dInfo):
        super().__init__()
        self.dim_input = dInfo['dim_input']
        self.dim_output = dInfo['dim_input']
        self.dim_hidden = dInfo['dim_hidden']
        self.num_layers = dInfo['num_layers']
        '''
          Fully connected MLP to predict the next state of the system
        '''
        self.deepnn = MLP([self.dim_input] + self.num_layers * [self.dim_hidden] + [self.dim_output])

        # Log the number of parameters
        num_params = sum(p.numel() for p in self.parameters())
        print('Number of embedding parameters UPDATED: {}'.format(num_params))

    def forward(self, x):
        z = self.deepnn(x)
        return z


'''
We define the model with an input layer, a number (defined) of intermediate
hidden layers, and an output layer. The output layer has de same dimmension than
the dimmensions of our problem (input dimmensions)
'''


class TINN_01(nn.Module):
    "Defines a MLP network"

    def __init__(self, dInfo):
        super().__init__()

        self.dim_input = dInfo['dim_input']
        self.dim_output = dInfo['dim_input']
        self.dim_hidden = dInfo['dim_hidden']
        self.num_layers = dInfo['num_layers']

        '''
        Components of the GENERIC formalism - Reversible systems
            L: L operator. Only off diagonal terms. L operator must be skew-symmetric.
            H: energy potential of the system.
        '''

        # construir la L aqui
        self.l_mat = torch.tensor([[0, 0, 1, 0],
                                   [0, 0, 0, 1],
                                   [-1, 0, 0, 0],
                                   [0, -1, 0, 0]], dtype=torch.float)

        self.energy = MLP([self.dim_input] + self.num_layers * [self.dim_hidden] + [self.dim_output])

    def forward(self, x):
        e = self.energy(x)
        l_mat_batch = self.l_mat.unsqueeze(0).expand(e.shape[0], -1, -1).to(e.device)
        dzdt = torch.bmm(l_mat_batch, e.unsqueeze(-1)).squeeze(-1)

        return dzdt


class TINN_02(nn.Module):
    "Defines a MLP network"

    def __init__(self, dInfo):
        super().__init__()
        self.dim_input = dInfo['dim_input']
        self.dim_output = dInfo['dim_input']
        self.dim_hidden = dInfo['dim_hidden']
        self.num_layers = dInfo['num_layers']

        self.dim_L_out = int(self.dim_output * (self.dim_output + 1) / 2 - self.dim_output)
        self.ones = torch.ones(self.dim_input, self.dim_input)

        '''
        Components of the GENERIC formalism - Reversible systems
            L: L operator. Only off diagonal terms. L operator must be skew-symmetric.
            H: energy potential of the system.
        '''

        self.energy_net = MLP([self.dim_input] + self.num_layers * [self.dim_hidden] + [self.dim_output])
        self.l_net = MLP([self.dim_input] + self.num_layers * [self.dim_hidden] + [self.dim_L_out])

    def forward(self, x):
        # Estimate the Energy gradient
        dedz = self.energy_net(x).unsqueeze(-1)

        # Estimate L operator componets
        L_out_vec = self.l_net(x)

        '''Reparametrization'''
        L_out = torch.zeros((x.size(0), self.dim_output, self.dim_output), device=L_out_vec.device)
        L_out[:, torch.tril(self.ones, -1) == 1] = L_out_vec
        # L - skew-symmetric
        L_out = (L_out - torch.transpose(L_out, 1, 2))

        #  Calculate the gradient
        dzdt = torch.bmm(L_out, dedz).squeeze(-1)

        return dzdt


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


class TINN_03(nn.Module):
    "Defines a MLP network"

    def __init__(self, dInfo):
        super().__init__()
        self.dim_input = dInfo['dim_input']
        self.dim_output = dInfo['dim_input']
        self.dim_hidden = dInfo['dim_hidden']
        self.num_layers = dInfo['num_layers']

        self.diag = torch.eye(self.dim_output, self.dim_output)
        self.diag = self.diag.reshape((-1, self.dim_output, self.dim_output))

        self.ones = torch.ones(self.dim_output, self.dim_output)

        self.dim_L_out = int(self.dim_output * (self.dim_output + 1) / 2) - self.dim_output
        self.dim_M_out = int(self.dim_output * (self.dim_output + 1) / 2)

        '''
        Components of the GENERIC formalism - Systems with dissipation
          L: L operator. Only off diagonal terms. L operator must be skew-symmetric.
          M: M operator. Compised by off diagonal and diagonal terms. M operator must be symmetric and possitive semidefinite.
          E: energy potential of the system.
          S: entropy potential of the system.
        '''

        # construir la L aqui
        # self.l_mat = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        #                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        #                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        #                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        #                            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                            [0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        #                            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
        #                            [0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
        #                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float)

        self.l_mat = torch.tensor([[0, 0, 1, 0],
                                   [0, 0, 0, 1],
                                   [-1, 0, 0, 0],
                                   [0, -1, 0, 0]], dtype=torch.float)

        self.m_net = MLP([self.dim_input] + self.num_layers * [self.dim_hidden] + [self.dim_M_out])

        self.energy_net = MLP([self.dim_input] + self.num_layers * [self.dim_hidden] + [self.dim_output])
        self.entropy_net = MLP([self.dim_input] + self.num_layers * [self.dim_hidden] + [self.dim_output])

    def forward(self, x):
        # Estimate the Energy gradient
        dedz = self.energy_net(x).unsqueeze(-1)

        # Estimate the Entropy gradient
        dsdz = self.entropy_net(x).unsqueeze(-1)

        L_out = self.l_mat.unsqueeze(0).expand(dedz.shape[0], -1, -1).to(dedz.device)

        # Estimate L operator componets
        # L_out_vec = self.l_net(x)

        # Estimate M operator componets
        M_out_vec = self.m_net(x)

        '''Reparametrization'''
        # L_out = torch.zeros((x.size(0), self.dim_output, self.dim_output), device=L_out_vec.device)
        # L_out[:, torch.tril(self.ones, -1) == 1] = L_out_vec
        # # L - skew-symmetric
        # L_out = (L_out - torch.transpose(L_out, 1, 2))

        M_out = torch.zeros((x.size(0), self.dim_output, self.dim_output), device=M_out_vec.device)

        M_out[:, torch.tril(self.ones) == 1] = M_out_vec
        # M - symmetric and possitive semidefinite
        self.diag = self.diag.to(M_out_vec.device)
        M_out = M_out - M_out * self.diag * abs(M_out) * self.diag  # Lower triangular
        M_out = torch.bmm(M_out, torch.transpose(M_out, 1, 2))  # Cholesky

        #  Calculate the gradient
        dzdt = torch.bmm(L_out, dedz) + torch.bmm(M_out, dsdz)

        deg_E = torch.bmm(M_out, dedz)
        deg_S = torch.bmm(L_out, dsdz)

        return dzdt.squeeze(-1), deg_E, deg_S



class TINN(nn.Module):
    "Defines a MLP network"

    def __init__(self, dInfo):
        super().__init__()
        self.dim_input = dInfo['dim_input']
        self.dim_output = dInfo['dim_input']
        self.dim_hidden = dInfo['dim_hidden']
        self.num_layers = dInfo['num_layers']

        self.diag = torch.eye(self.dim_output, self.dim_output)
        self.diag = self.diag.reshape((-1, self.dim_output, self.dim_output))

        self.ones = torch.ones(self.dim_output, self.dim_output)

        self.dim_L_out = int(self.dim_output * (self.dim_output + 1) / 2) - self.dim_output
        self.dim_M_out = int(self.dim_output * (self.dim_output + 1) / 2)

        '''
        Components of the GENERIC formalism - Systems with dissipation
          L: L operator. Only off diagonal terms. L operator must be skew-symmetric.
          M: M operator. Compised by off diagonal and diagonal terms. M operator must be symmetric and possitive semidefinite.
          E: energy potential of the system.
          S: entropy potential of the system.
        '''

        self.l_net = MLP([self.dim_input] + self.num_layers * [self.dim_hidden] + [self.dim_L_out])
        self.m_net = MLP([self.dim_input] + self.num_layers * [self.dim_hidden] + [self.dim_M_out])

        self.energy_net = MLP([self.dim_input] + self.num_layers * [self.dim_hidden] + [self.dim_output])
        self.entropy_net = MLP([self.dim_input] + self.num_layers * [self.dim_hidden] + [self.dim_output])

    def forward(self, x):
        # Estimate the Energy gradient
        dedz = self.energy_net(x).unsqueeze(-1)

        # Estimate the Entropy gradient
        dsdz = self.entropy_net(x).unsqueeze(-1)

        # Estimate L operator componets
        L_out_vec = self.l_net(x)

        # Estimate M operator componets
        M_out_vec = self.m_net(x)

        '''Reparametrization'''
        L_out = torch.zeros((x.size(0), self.dim_output, self.dim_output), device=L_out_vec.device)
        L_out[:, torch.tril(self.ones, -1) == 1] = L_out_vec
        # L - skew-symmetric
        L_out = (L_out - torch.transpose(L_out, 1, 2))

        M_out = torch.zeros((x.size(0), self.dim_output, self.dim_output), device=M_out_vec.device)

        M_out[:, torch.tril(self.ones) == 1] = M_out_vec
        # M - symmetric and possitive semidefinite
        self.diag = self.diag.to(M_out_vec.device)
        M_out = M_out - M_out * self.diag * abs(M_out) * self.diag  # Lower triangular
        M_out = torch.bmm(M_out, torch.transpose(M_out, 1, 2))  # Cholesky

        #  Calculate the gradient
        dzdt = torch.bmm(L_out, dedz) + torch.bmm(M_out, dsdz)

        deg_E = torch.bmm(M_out, dedz)
        deg_S = torch.bmm(L_out, dsdz)

        return dzdt.squeeze(-1), deg_E, deg_S
