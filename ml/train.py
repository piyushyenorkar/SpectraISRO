"""
SPECTRA — Full pix2pix GAN Training Script

Usage:
    python train.py                           # Train from scratch
    python train.py --resume checkpoints/epoch_100.pth  # Resume from checkpoint

Output:
    - checkpoints/epoch_XXX.pth   — model checkpoints
    - checkpoints/generator_best.pth — best generator weights
    - samples/epoch_XXXX.png     — visual samples each epoch
"""

import os
import argparse
import time

import torch
from torch.utils.data import DataLoader

import config
from dataset import IRColorDataset
from models.generator import UNetGenerator, init_weights
from models.discriminator import PatchGANDiscriminator
from losses import GeneratorLoss, DiscriminatorLoss
from utils import save_checkpoint, load_checkpoint, save_sample_images
from evaluate import compute_psnr_ssim


def train():
    parser = argparse.ArgumentParser(description="SPECTRA pix2pix Training")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--data", type=str, default=config.DATA_ROOT, help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()

    # ─── Device ────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ─── Dataset ───────────────────────────────────────────
    train_dataset = IRColorDataset(root=args.data, split="train", img_size=config.IMG_SIZE, jitter_size=config.JITTER_SIZE)
    val_dataset = IRColorDataset(root=args.data, split="val", img_size=config.IMG_SIZE, jitter_size=config.IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    print(f"📦 Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")

    # ─── Models ────────────────────────────────────────────
    generator = UNetGenerator(
        in_channels=config.INPUT_CHANNELS,
        out_channels=config.OUTPUT_CHANNELS,
        base_filters=config.GENERATOR_FILTERS,
    ).to(device)

    discriminator = PatchGANDiscriminator(
        in_channels=config.INPUT_CHANNELS + config.OUTPUT_CHANNELS,
        base_filters=config.DISCRIMINATOR_FILTERS,
    ).to(device)

    # Initialize weights
    init_weights(generator)
    init_weights(discriminator)

    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"🧠 Generator:     {gen_params:>10,} params")
    print(f"🧠 Discriminator: {disc_params:>10,} params")

    # ─── Optimizers ────────────────────────────────────────
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(config.BETA1, config.BETA2))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(config.BETA1, config.BETA2))

    # ─── Losses ────────────────────────────────────────────
    gen_criterion = GeneratorLoss(lambda_l1=config.LAMBDA_L1)
    disc_criterion = DiscriminatorLoss()

    # ─── Resume ────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, generator, discriminator, opt_g, opt_d, device)
        print(f"▶️  Resumed from epoch {start_epoch}")

    # ─── Training Loop ─────────────────────────────────────
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.SAMPLE_DIR, exist_ok=True)

    best_psnr = 0.0
    print(f"\n{'='*60}")
    print(f"🚀 Starting training for {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        generator.train()
        discriminator.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_l1_loss = 0.0
        start_time = time.time()

        for batch_idx, (ir_images, rgb_images) in enumerate(train_loader):
            ir_images = ir_images.to(device)
            rgb_images = rgb_images.to(device)

            # ─── Train Discriminator ───────────────────────
            opt_d.zero_grad()

            # Generate fake RGB
            fake_rgb = generator(ir_images)

            # Discriminator predictions
            disc_real = discriminator(ir_images, rgb_images)
            disc_fake = discriminator(ir_images, fake_rgb.detach())

            # Discriminator loss
            d_loss, d_real_loss, d_fake_loss = disc_criterion(disc_real, disc_fake)
            d_loss.backward()
            opt_d.step()

            # ─── Train Generator ──────────────────────────
            opt_g.zero_grad()

            # Re-evaluate discriminator on fake (need fresh gradients)
            disc_fake_for_g = discriminator(ir_images, fake_rgb)

            # Generator loss
            g_loss, g_gan_loss, g_l1_loss = gen_criterion(disc_fake_for_g, fake_rgb, rgb_images)
            g_loss.backward()
            opt_g.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_l1_loss += g_l1_loss.item()

        # ─── Epoch Stats ──────────────────────────────────
        n_batches = len(train_loader)
        elapsed = time.time() - start_time
        avg_g = epoch_g_loss / n_batches
        avg_d = epoch_d_loss / n_batches
        avg_l1 = epoch_l1_loss / n_batches

        print(
            f"Epoch [{epoch+1:3d}/{args.epochs}] "
            f"G_loss: {avg_g:.4f} | D_loss: {avg_d:.4f} | L1: {avg_l1:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # ─── Save Samples ─────────────────────────────────
        generator.eval()
        with torch.inference_mode():
            val_ir, val_rgb = next(iter(val_loader))
            val_ir = val_ir.to(device)
            val_rgb = val_rgb.to(device)
            val_fake = generator(val_ir)
            save_sample_images(val_ir, val_rgb, val_fake, epoch + 1, config.SAMPLE_DIR)

        # ─── Validation Metrics ────────────────────────────
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            avg_psnr, avg_ssim = validate(generator, val_loader, device)
            print(f"  📊 Val PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                # Save best generator weights only
                best_path = os.path.join(config.CHECKPOINT_DIR, "generator_best.pth")
                torch.save(generator.state_dict(), best_path)
                print(f"  🏆 New best! PSNR: {best_psnr:.2f} dB → saved to {best_path}")

        # ─── Save Checkpoint ──────────────────────────────
        if (epoch + 1) % config.SAVE_EVERY == 0:
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"epoch_{epoch+1:04d}.pth")
            save_checkpoint(generator, discriminator, opt_g, opt_d, epoch + 1, ckpt_path)

    # ─── Final Save ────────────────────────────────────────
    final_path = os.path.join(config.CHECKPOINT_DIR, "generator_final.pth")
    torch.save(generator.state_dict(), final_path)
    print(f"\n✅ Training complete! Best PSNR: {best_psnr:.2f} dB")
    print(f"   Best weights:  {os.path.join(config.CHECKPOINT_DIR, 'generator_best.pth')}")
    print(f"   Final weights: {final_path}")


def validate(generator, val_loader, device):
    """Compute average PSNR and SSIM on validation set."""
    generator.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.inference_mode():
        for ir_images, rgb_images in val_loader:
            ir_images = ir_images.to(device)
            rgb_images = rgb_images.to(device)
            fake_rgb = generator(ir_images)

            psnr, ssim = compute_psnr_ssim(fake_rgb, rgb_images)
            total_psnr += psnr
            total_ssim += ssim
            count += 1

    return total_psnr / count, total_ssim / count


if __name__ == "__main__":
    train()
