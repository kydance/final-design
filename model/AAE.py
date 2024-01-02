import torch
import torch.nn.functional as F
import numpy as np


class AE(torch.nn.Module):
    def __init__(self, dim_data, dim_z, n_hidden=400, n_hidden_d=1000, n_output_d=1):
        super(AE, self).__init__()
        self.dim_data = dim_data
        self.dim_z = dim_z
        self.n_hidden = n_hidden

        # MLP_encoder
        self.encoder1 = torch.nn.Linear(self.dim_data, self.n_hidden)
        self.encoder2 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.encoder3 = torch.nn.Linear(self.n_hidden, self.dim_z)
        # MLP_decoker
        self.decoder1 = torch.nn.Linear(self.dim_z, self.n_hidden)
        self.decoder2 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.decoder3 = torch.nn.Linear(self.n_hidden, self.dim_data)


    def MLP_encoder(self, x):
        x1 = self.encoder1(x)
        x1 = F.leaky_relu(x1, 0.1, inplace=False)

        x2 = self.encoder2(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=False)

        z = self.encoder3(x2)
        return z

    def MLP_decoder(self, z):
        z1 = self.decoder1(z)
        z1 = F.leaky_relu(z1, 0.1, inplace=False)

        z2 = self.decoder2(z1)
        z2 = F.leaky_relu(z2, 0.1, inplace=False)

        y = self.decoder3(z2)
        y = F.tanh(y)
        return y

    def my_loss(self, y_true, y_pred, d):
        d = torch.transpose(d, 1, 0)
        A = torch.sum(torch.multiply(y_pred, d), dim=1)
        B = torch.norm(y_pred, p=2, dim=1)
        C = torch.norm(d, p=2)
        defen = torch.div(A, B*C+1e-5)
        s = torch.topk(defen, k=20, dim=0).values
        sam_loss = torch.sum(s)
        mse_loss = F.mse_loss(y_pred, y_true, reduce=True)
        distance_loss = mse_loss + 0.1 * sam_loss

        return distance_loss

    def forward(self, x, data_input, with_decoder=True):
        z = self.MLP_encoder(x)
        if with_decoder == False:
            return z

        y = self.MLP_decoder(z)
        # loss
        R_loss = torch.mean(torch.mean(self.my_loss(x, y, data_input)))

        R_loss = torch.mean(R_loss)

        return y, z, R_loss


class GAN(torch.nn.Module):
    def __init__(self, dim_data, dim_z, n_hidden_d=1000, n_output_d=1):
        super(GAN, self).__init__()
        self.dim_data = dim_data
        self.dim_z = dim_z
        self.n_hidden_d = n_hidden_d
        self.n_output_d = n_output_d
        # discriminator_real
        self.dr1 = torch.nn.Linear(self.dim_z, self.n_hidden_d)
        self.dr2 = torch.nn.Linear(self.n_hidden_d, self.n_hidden_d)
        self.dr3 = torch.nn.Linear(self.n_hidden_d, self.n_output_d)
        # discriminator_fake
        self.df1 = torch.nn.Linear(self.dim_z, self.n_hidden_d)
        self.df2 = torch.nn.Linear(self.n_hidden_d, self.n_hidden_d)
        self.df3 = torch.nn.Linear(self.n_hidden_d, self.n_output_d)

    def discriminator_real(self, z):
        z1 = self.dr1(z)
        z1 = F.leaky_relu(z1, inplace=False)

        z2 = self.dr2(z1)
        z2 = F.leaky_relu(z2, inplace=False)

        y = self.dr3(z2)
        return torch.sigmoid(y), y

    def discriminator_fake(self, z):
        z1 = self.df1(z)
        z1 = F.leaky_relu(z1, inplace=False)

        z2 = self.df2(z1)
        z2 = F.leaky_relu(z2, inplace=False)

        y = self.df3(z2)
        return torch.sigmoid(y), y

    def forward(self, z, with_G=True):
        z_simples = np.random.randn(self.dim_data, self.dim_z)
        z_simples = torch.tensor(z_simples, dtype=torch.float32).to(z.device)
        z_real = z_simples
        z_fake = z
        D_real, D_real_logits = self.discriminator_real(z_real)
        D_fake, D_fake_logits = self.discriminator_fake(z_fake)
        # D_real_logits.requires_grad = False
        # D_fake_logits.requires_grad = False
        # discriminator loss
        D_loss_real = torch.mean(F.binary_cross_entropy_with_logits(D_real_logits, torch.ones_like(D_real_logits)))
        D_loss_fake = torch.mean(F.binary_cross_entropy_with_logits(D_fake_logits, torch.zeros_like(D_fake_logits)))
        D_loss = 0.5 * (D_loss_real + D_loss_fake)
        # generator loss
        G_loss = torch.mean(F.binary_cross_entropy_with_logits(D_fake_logits, torch.ones_like(D_fake_logits)))

        if with_G == False:
            D_loss = torch.mean(D_loss)
            return D_loss
        D_loss = torch.mean(D_loss)
        G_loss = torch.mean(G_loss)
        return D_loss, G_loss