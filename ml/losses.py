"""
SPECTRA — Loss Functions for pix2pix GAN Training
"""

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """
    Adversarial loss for pix2pix.
    Uses MSE (LSGAN) for more stable training than BCE.
    """

    def __init__(self, use_lsgan: bool = True):
        super().__init__()
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        if is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        return self.loss(prediction, target)


class GeneratorLoss(nn.Module):
    """
    Combined generator loss: adversarial + L1 reconstruction.

    L_G = L_GAN(G(x), 1) + λ * L1(G(x), y)
    """

    def __init__(self, lambda_l1: float = 100.0, use_lsgan: bool = True):
        super().__init__()
        self.gan_loss = GANLoss(use_lsgan=use_lsgan)
        self.l1_loss = nn.L1Loss()
        self.lambda_l1 = lambda_l1

    def forward(
        self,
        disc_fake_output: torch.Tensor,
        generated: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss, gan_loss, l1_loss
        """
        loss_gan = self.gan_loss(disc_fake_output, is_real=True)
        loss_l1 = self.l1_loss(generated, target)
        total = loss_gan + self.lambda_l1 * loss_l1
        return total, loss_gan, loss_l1


class DiscriminatorLoss(nn.Module):
    """
    Discriminator loss: average of real and fake classification losses.

    L_D = 0.5 * [L_GAN(D(x, y), 1) + L_GAN(D(x, G(x)), 0)]
    """

    def __init__(self, use_lsgan: bool = True):
        super().__init__()
        self.gan_loss = GANLoss(use_lsgan=use_lsgan)

    def forward(
        self,
        disc_real_output: torch.Tensor,
        disc_fake_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss, real_loss, fake_loss
        """
        loss_real = self.gan_loss(disc_real_output, is_real=True)
        loss_fake = self.gan_loss(disc_fake_output, is_real=False)
        total = 0.5 * (loss_real + loss_fake)
        return total, loss_real, loss_fake
