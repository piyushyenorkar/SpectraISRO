"""
SPECTRA — Multi-Mode Training Script for ISRO PS10

Supports three training modes:
    --mode sr        Train Model A (Super-Resolution: 200m→100m TIR)
    --mode colorize  Train Model B (Colorization: 100m TIR→100m RGB)
    --mode e2e       Train SpectraSatNet end-to-end (both stages chained)

Usage:
    python train.py --mode sr                              # Train SR model
    python train.py --mode colorize                        # Train colorizer
    python train.py --mode e2e                             # Train end-to-end
    python train.py --mode sr --resume checkpoints/sr_epoch_100.pth  # Resume

Output:
    - checkpoints/{mode}_epoch_XXX.pth    — model checkpoints
    - checkpoints/{mode}_best.pth         — best model weights
    - samples/{mode}_epoch_XXXX.png       — visual samples
"""

import os
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import LandsatSRDataset, LandsatColorDataset
from model import SpectraSatNet, thermal_emissivity_loss
from models.generator import UNetGenerator, init_weights
from models.discriminator import PatchGANDiscriminator
from losses import GeneratorLoss, DiscriminatorLoss
from utils import save_checkpoint, load_checkpoint, save_sample_images
from evaluate import compute_psnr_ssim


def train_sr(args):
    """Train Model A — Super-Resolution (200m→100m TIR)."""
    print("\n🔬 Mode: SUPER-RESOLUTION (Model A)")
    print("   Input:  200m TIR (grayscale)")
    print("   Output: 100m TIR (grayscale)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    # Dataset
    train_ds = LandsatSRDataset(root=config.SR_DATA_ROOT, split="train",
                                 img_size=config.IMG_SIZE, jitter_size=config.JITTER_SIZE)
    val_ds = LandsatSRDataset(root=config.SR_DATA_ROOT, split="val",
                               img_size=config.IMG_SIZE, jitter_size=config.IMG_SIZE)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)
    
    print(f"   Train: {len(train_ds)} patches | Val: {len(val_ds)} patches")
    
    # Model: Simple SR network (just the SR branch of SpectraSatNet)
    model = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=5, padding=2),
        nn.PReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.PReLU(),
        nn.Conv2d(64, 1, kernel_size=3, padding=1),
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {params:,}")
    
    # Loss & Optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(config.BETA1, config.BETA2))
    
    # Resume
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"   Resumed from epoch {start_epoch}")
    
    # Training
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.SAMPLE_DIR, exist_ok=True)
    best_psnr = 0.0
    
    print(f"\n{'='*60}")
    print(f"🚀 Training SR for {args.epochs} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        for low_res, high_res in train_loader:
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            
            optimizer.zero_grad()
            output = model(low_res)
            loss = criterion(output, high_res)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        elapsed = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1:3d}/{args.epochs}] Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s")
        
        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            total_psnr, total_ssim, count = 0.0, 0.0, 0
            with torch.inference_mode():
                for low_res, high_res in val_loader:
                    low_res = low_res.to(device)
                    high_res = high_res.to(device)
                    output = model(low_res)
                    # Expand to 3-channel for PSNR/SSIM computation
                    psnr, ssim = compute_psnr_ssim(output.repeat(1, 3, 1, 1), high_res.repeat(1, 3, 1, 1))
                    total_psnr += psnr
                    total_ssim += ssim
                    count += 1
            
            avg_psnr = total_psnr / count
            avg_ssim = total_ssim / count
            print(f"  📊 Val PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
            
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "sr_best.pth"))
                print(f"  🏆 New best! PSNR: {best_psnr:.2f} dB")
        
        # Checkpoint
        if (epoch + 1) % config.SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(config.CHECKPOINT_DIR, f"sr_epoch_{epoch+1:04d}.pth"))
    
    print(f"\n✅ SR Training complete! Best PSNR: {best_psnr:.2f} dB")


def train_colorize(args):
    """Train Model B — Colorization (100m TIR → 100m RGB) using pix2pix."""
    print("\n🎨 Mode: COLORIZATION (Model B)")
    print("   Input:  100m TIR (grayscale)")
    print("   Output: 100m RGB (color)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    # Dataset
    train_ds = LandsatColorDataset(root=config.COLOR_DATA_ROOT, split="train",
                                    img_size=config.IMG_SIZE, jitter_size=config.JITTER_SIZE)
    val_ds = LandsatColorDataset(root=config.COLOR_DATA_ROOT, split="val",
                                  img_size=config.IMG_SIZE, jitter_size=config.IMG_SIZE)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)
    
    print(f"   Train: {len(train_ds)} patches | Val: {len(val_ds)} patches")
    
    # Models
    generator = UNetGenerator(
        in_channels=config.INPUT_CHANNELS,
        out_channels=config.OUTPUT_CHANNELS,
        base_filters=config.GENERATOR_FILTERS,
    ).to(device)
    
    discriminator = PatchGANDiscriminator(
        in_channels=config.INPUT_CHANNELS + config.OUTPUT_CHANNELS,
        base_filters=config.DISCRIMINATOR_FILTERS,
    ).to(device)
    
    init_weights(generator)
    init_weights(discriminator)
    
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"   Generator:     {gen_params:>10,} params")
    print(f"   Discriminator: {disc_params:>10,} params")
    
    # Optimizers
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(config.BETA1, config.BETA2))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(config.BETA1, config.BETA2))
    
    # Losses
    gen_criterion = GeneratorLoss(lambda_l1=config.LAMBDA_L1)
    disc_criterion = DiscriminatorLoss()
    
    # Resume
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, generator, discriminator, opt_g, opt_d, device)
        print(f"   Resumed from epoch {start_epoch}")
    
    # Training
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.SAMPLE_DIR, exist_ok=True)
    best_psnr = 0.0
    
    print(f"\n{'='*60}")
    print(f"🚀 Training Colorizer for {args.epochs} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.epochs):
        generator.train()
        discriminator.train()
        epoch_g_loss, epoch_d_loss, epoch_l1_loss = 0.0, 0.0, 0.0
        start_time = time.time()
        
        for ir_images, rgb_images in train_loader:
            ir_images = ir_images.to(device)
            rgb_images = rgb_images.to(device)
            
            # Train Discriminator
            opt_d.zero_grad()
            fake_rgb = generator(ir_images)
            disc_real = discriminator(ir_images, rgb_images)
            disc_fake = discriminator(ir_images, fake_rgb.detach())
            d_loss, _, _ = disc_criterion(disc_real, disc_fake)
            d_loss.backward()
            opt_d.step()
            
            # Train Generator
            opt_g.zero_grad()
            disc_fake_for_g = discriminator(ir_images, fake_rgb)
            g_loss, g_gan_loss, g_l1_loss = gen_criterion(disc_fake_for_g, fake_rgb, rgb_images)
            g_loss.backward()
            opt_g.step()
            
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_l1_loss += g_l1_loss.item()
        
        n = len(train_loader)
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1:3d}/{args.epochs}] "
              f"G: {epoch_g_loss/n:.4f} | D: {epoch_d_loss/n:.4f} | "
              f"L1: {epoch_l1_loss/n:.4f} | {elapsed:.1f}s")
        
        # Save samples
        generator.eval()
        with torch.inference_mode():
            val_ir, val_rgb = next(iter(val_loader))
            val_ir, val_rgb = val_ir.to(device), val_rgb.to(device)
            val_fake = generator(val_ir)
            save_sample_images(val_ir, val_rgb, val_fake, epoch + 1,
                             os.path.join(config.SAMPLE_DIR, "colorize"))
        
        # Validation
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            avg_psnr, avg_ssim = validate_colorize(generator, val_loader, device)
            print(f"  📊 Val PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
            
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save(generator.state_dict(),
                          os.path.join(config.CHECKPOINT_DIR, "colorize_best.pth"))
                print(f"  🏆 New best! PSNR: {best_psnr:.2f} dB")
        
        # Checkpoint
        if (epoch + 1) % config.SAVE_EVERY == 0:
            save_checkpoint(generator, discriminator, opt_g, opt_d, epoch + 1,
                          os.path.join(config.CHECKPOINT_DIR, f"colorize_epoch_{epoch+1:04d}.pth"))
    
    print(f"\n✅ Colorization Training complete! Best PSNR: {best_psnr:.2f} dB")


def train_e2e(args):
    """Train SpectraSatNet end-to-end (SR + Colorization chained)."""
    print("\n🔗 Mode: END-TO-END (SpectraSatNet)")
    print("   Input:  200m TIR (grayscale)")
    print("   Output: 100m TIR + 100m RGB")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    # We need BOTH datasets loaded simultaneously for e2e training
    # SR dataset provides 200m→100m pairs, Color dataset provides 100m TIR→RGB pairs
    sr_train = LandsatSRDataset(root=config.SR_DATA_ROOT, split="train",
                                 img_size=config.IMG_SIZE, jitter_size=config.JITTER_SIZE)
    color_train = LandsatColorDataset(root=config.COLOR_DATA_ROOT, split="train",
                                       img_size=config.IMG_SIZE, jitter_size=config.JITTER_SIZE)
    
    sr_loader = DataLoader(sr_train, batch_size=config.BATCH_SIZE, shuffle=True,
                           num_workers=2, pin_memory=True)
    color_loader = DataLoader(color_train, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    
    # Model
    model = SpectraSatNet().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {params:,}")
    
    # Losses
    sr_criterion = nn.L1Loss()      # For SR output
    color_criterion = nn.L1Loss()   # For colorization output
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(config.BETA1, config.BETA2))
    
    # Resume
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"   Resumed from epoch {start_epoch}")
    
    # Training
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.SAMPLE_DIR, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🚀 Training End-to-End for {args.epochs} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_sr_loss, epoch_color_loss, epoch_physics_loss = 0.0, 0.0, 0.0
        start_time = time.time()
        
        # Iterate over both loaders
        color_iter = iter(color_loader)
        for batch_idx, (low_res_tir, high_res_tir) in enumerate(sr_loader):
            low_res_tir = low_res_tir.to(device)
            high_res_tir = high_res_tir.to(device)
            
            # Get corresponding color pair
            try:
                _, target_rgb = next(color_iter)
            except StopIteration:
                color_iter = iter(color_loader)
                _, target_rgb = next(color_iter)
            target_rgb = target_rgb.to(device)
            
            # Forward pass through unified model
            optimizer.zero_grad()
            out_sr, out_rgb = model(low_res_tir)
            
            # SR Loss (compare super-resolved TIR with ground truth 100m TIR)
            loss_sr = sr_criterion(out_sr, high_res_tir) * config.LAMBDA_SR
            
            # Colorization Loss (compare colorized RGB with ground truth RGB)
            loss_color = color_criterion(out_rgb, target_rgb) * config.LAMBDA_L1
            
            # Physics-Informed Loss (bonus)
            loss_physics = thermal_emissivity_loss(out_sr, out_rgb) * config.LAMBDA_PHYSICS
            
            # Total loss
            total_loss = loss_sr + loss_color + loss_physics
            total_loss.backward()
            optimizer.step()
            
            epoch_sr_loss += loss_sr.item()
            epoch_color_loss += loss_color.item()
            epoch_physics_loss += loss_physics.item()
        
        n = len(sr_loader)
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1:3d}/{args.epochs}] "
              f"SR: {epoch_sr_loss/n:.4f} | Color: {epoch_color_loss/n:.4f} | "
              f"Physics: {epoch_physics_loss/n:.4f} | {elapsed:.1f}s")
        
        # Checkpoint
        if (epoch + 1) % config.SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(config.CHECKPOINT_DIR, f"e2e_epoch_{epoch+1:04d}.pth"))
    
    # Final save
    torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "e2e_best.pth"))
    print(f"\n✅ End-to-End Training complete!")


def validate_colorize(generator, val_loader, device):
    """Compute average PSNR/SSIM on colorization validation set."""
    generator.eval()
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    with torch.inference_mode():
        for ir, rgb in val_loader:
            ir, rgb = ir.to(device), rgb.to(device)
            fake = generator(ir)
            psnr, ssim = compute_psnr_ssim(fake, rgb)
            total_psnr += psnr
            total_ssim += ssim
            count += 1
    return total_psnr / count, total_ssim / count


def main():
    parser = argparse.ArgumentParser(description="SPECTRA — Multi-Mode Training (ISRO PS10)")
    parser.add_argument("--mode", type=str, required=True, choices=["sr", "colorize", "e2e"],
                        help="Training mode: sr | colorize | e2e")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--data", type=str, default=None, help="Override data root directory")
    args = parser.parse_args()

    print("=" * 60)
    print("SPECTRA — Two-Stage Training Pipeline (ISRO PS10)")
    print("=" * 60)

    if args.mode == "sr":
        train_sr(args)
    elif args.mode == "colorize":
        train_colorize(args)
    elif args.mode == "e2e":
        train_e2e(args)


if __name__ == "__main__":
    main()
