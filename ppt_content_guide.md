# SPECTRA — PPT Ideation Submission Guide

## Part 1: What ISRO Actually Wants (From the Explainer Session)

Based on a thorough analysis of all 6 screenshots from the explainer session, ISRO's expectations are crystal clear:

### The Exact Problem
- **Input:** Single-channel TIR (Thermal Infrared) image @ 200m resolution
- **Output 1:** High-resolution TIR image @ 100m (super-resolution)
- **Output 2:** Colorized RGB image @ 100m (IR colorization)

### The Exact Workflow (ISRO Slide 6 — Workflow Summary)
```
Start: Data Source
  → Download Landsat 9 Bands: B2, B3, B4, B10
  → Three parallel paths:
      Path 1: Merge B2+B3+B4 into RGB → Downscale RGB by 3.3x to 100m → Create patches (ground truth for colorization)
      Path 2: Downscale TIR B10 by 3.3x to 100m (ground truth for SR)
      Path 3: Downscale TIR B10 by 6.7x to 200m → Create patches (model input)
  → Super-resolution model (200m → 100m)
  → IR Image Colorization
  → End: Colorized IR Image
```

### The Exact Evaluation Criteria (ISRO Slide 7)
| Category | Metric |
|----------|--------|
| **Image Quality** | PSNR, SSIM, FID |
| **Qualitative** | Visual inspection to prevent "hallucinations" |
| **Preferred** | Low inference time per tile |
| **Bonus** | **Physics-informed modeling** |

### The Exact Tech Stack ISRO Expects (Slide 5)
- **Dataset:** Landsat 9 (USGS) processed TIR1-RGB pairs using their GitHub instructions
- **Framework:** Python-based end-to-end solution
- **Models:** GANs / Diffusion Models / SoTA Image-to-Image models
- **Libraries:** GDAL, Rasterio, tifffile, OpenCV

### What the Ideation Submission Must Answer (PPT Template — Slide 0)
1. How different is it from any of the other existing ideas?
2. How will it be able to solve the problem?
3. USP of the proposed solution

---

## Part 2: Honest Assessment — Are We on the Right Path?

### ✅ What We Got RIGHT (Perfectly Aligned with ISRO)

| ISRO Requirement | Our Implementation | Status |
|---|---|---|
| Landsat 9 B2, B3, B4, B10 data | ✅ We download exactly these 4 bands via GEE | ✅ Perfect |
| Merge B2+B3+B4 into RGB | ✅ `preprocess_landsat.py` does this | ✅ Perfect |
| Downscale RGB to 100m (3.3x) | ✅ Done in preprocessing | ✅ Perfect |
| Downscale TIR to 100m (3.3x) for GT | ✅ Done in preprocessing | ✅ Perfect |
| Downscale TIR to 200m (6.7x) for input | ✅ Done in preprocessing | ✅ Perfect |
| Two-stage pipeline (SR → Colorize) | ✅ `SpectraSatNet` has both stages | ✅ Perfect |
| PSNR + SSIM evaluation | ✅ `evaluate.py` computes both | ✅ Perfect |
| Physics-informed loss (BONUS) | ✅ `thermal_emissivity_loss()` in model.py | ✅ Perfect |
| Python-based end-to-end | ✅ PyTorch, Rasterio, OpenCV | ✅ Perfect |
| GAN architecture | ✅ pix2pix (UNet Generator + PatchGAN) | ✅ Perfect |
| Diverse Indian terrain data | ✅ 8 regions covering all terrain types | ✅ Perfect |

### ⚠️ What to Be Careful About in the PPT

1. **Do NOT over-emphasize the web frontend.** ISRO wants a Deep Learning pipeline, not a web app. The web frontend is nice but secondary. In the PPT, the focus should be on the ML architecture and approach.

2. **Mention you will use their dataset/scripts when provided.** They said they'll share preprocessing scripts. In your PPT, say you will use their provided pipeline but show you already understand the exact workflow.

3. **FID metric is missing.** ISRO asked for PSNR, SSIM, and FID. We have the first two but not FID (Fréchet Inception Distance). Mention it in the PPT approach.

### 🎯 Shortlisting Verdict: YES, This Approach Can Get Shortlisted

**Why I believe this will work:**
- You follow ISRO's exact 4-step workflow (Input 200m → SR to 100m → Colorize → Compare)
- You use the exact dataset they specified (Landsat 9)
- You have the exact bands they specified (B2, B3, B4, B10)
- You have a two-stage deep learning pipeline which is precisely what they asked for
- You implemented the bonus **physics-informed modeling** that they specifically called out
- Your evaluation covers PSNR + SSIM which they explicitly listed
- You trained on diverse Indian terrain (8 regions) which shows domain awareness

**What will make you stand out from other teams:**
- Most teams will propose a single-stage approach. Your two-stage (SR + Colorization) with end-to-end fine-tuning is architecturally superior
- The physics-informed thermal emissivity loss is the BONUS they asked for — most teams will skip this
- Having a working prototype with actual Landsat 9 data already processed (even if they said don't worry about data yet) shows initiative
- The ONNX export for low-latency inference directly addresses their "low inference time per tile" preference

---

## Part 3: Slide-by-Slide PPT Content

> [!IMPORTANT]
> The PPT template has 7 slides. Below is exactly what to write in each.

---

### SLIDE 1: Title Slide (Already provided by ISRO template — just add your info)

**Title:** SPECTRA — Two-Stage Physics-Informed Pipeline for IR Image Colorization & Enhancement

**Team Name:** Team STARCY

**Team Members:** Piyush Yenorkar (Team Lead) + [other members]

**Problem Statement:** PS10 — Infrared Image Colorization and Enhancement for Improved Object Interpretation

---

### SLIDE 2: Opportunity — How different? How will it solve? USP?

**Content to write:**

**How Different From Existing Ideas:**

Most existing IR colorization approaches use a single-stage model that directly maps grayscale TIR to RGB. This ignores the fundamental resolution mismatch — TIR sensors capture at 100m native resolution while optical sensors capture at 30m. Our approach addresses this with a dedicated two-stage pipeline.

**How It Solves the Problem:**

SPECTRA uses a cascaded deep learning pipeline:
- **Stage 1 — Super-Resolution:** A Sub-Pixel Convolution Network upscales the 200m TIR input to 100m, recovering spatial details lost due to coarser sensor resolution.
- **Stage 2 — Colorization:** A pix2pix GAN (U-Net Generator + PatchGAN Discriminator) maps the enhanced 100m TIR to 100m RGB, guided by learned spectral correlations between thermal and visible bands.
- **End-to-End Fine-Tuning:** Both stages are jointly optimized via a unified `SpectraSatNet` architecture with skip connections, ensuring the SR output is optimized for the colorization task.

**USP of the Proposed Solution:**
1. **Two-Stage Architecture:** Dedicated SR + Colorization stages outperform monolithic models
2. **Physics-Informed Loss:** A custom thermal emissivity constraint ensures thermodynamic consistency (e.g., hot pixels → urban/bare soil colors, cold pixels → vegetation/water colors), reducing hallucinations
3. **Low Inference Time:** Lightweight architecture (~1.5M parameters for end-to-end model) enables fast per-tile processing, exportable to ONNX for optimized deployment
4. **Indian Terrain Diversity:** Trained on 8 geographically diverse Indian regions (urban, desert, snow, delta, coastal, rural, water bodies, mixed) ensuring generalization across ISRO's target scenarios

---

### SLIDE 3: List of Features Offered by the Solution

**Content to write (with visual icons/bullets):**

**Core ML Pipeline Features:**
- ⚡ Two-stage pipeline: Super-Resolution (200m→100m) + Colorization (TIR→RGB)
- 🧠 End-to-end trainable unified model (`SpectraSatNet`) with shared feature representations
- 🔬 Physics-informed thermal emissivity loss for thermodynamically consistent colorization
- 📊 Quantitative evaluation: PSNR, SSIM, and FID metrics on validation set
- 🖼️ Visual inspection module for hallucination detection
- 📦 ONNX model export for deployment-ready, low-latency inference

**Data Pipeline Features:**
- 🛰️ Automated Landsat 9 data acquisition (B2, B3, B4, B10 bands)
- 🔄 ISRO-aligned preprocessing: RGB merge, 3.3x/6.7x downscaling, patch extraction
- 🌍 Geographically diverse training data across 8 Indian terrain categories
- 📐 Configurable patch sizes with overlapping stride for dense training

**Deployment Features:**
- 🚀 ONNX Runtime inference (low inference time per tile)
- 🌐 Optional web-based visualization tool for side-by-side comparison (IR vs Enhanced vs Ground Truth)
- 📈 Automated metric reporting (PSNR/SSIM per tile and average)

> [!TIP]
> Add a small visual: a 2x2 grid showing (1) 200m TIR input, (2) 100m SR output, (3) Colorized RGB output, (4) Ground truth RGB. You can use screenshots from your actual training samples.

---

### SLIDE 4: Process Flow Diagram

**Create a clean flowchart with these boxes and arrows:**

```
┌─────────────────────┐
│   Landsat 9 Data    │
│  (B2, B3, B4, B10)  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Preprocessing     │
│ • RGB merge (B2+B3+B4)            │
│ • Downscale RGB → 100m (3.3x)     │
│ • Downscale TIR → 100m (3.3x) [GT]│
│ • Downscale TIR → 200m (6.7x) [Input] │
│ • Patch extraction (128×128)       │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│         SpectraSatNet (Training)        │
│                                         │
│  ┌──────────────┐    ┌───────────────┐  │
│  │  Stage 1: SR │───→│ Stage 2: Color│  │
│  │  200m → 100m │    │ TIR → RGB     │  │
│  │  (SubPixel   │    │ (pix2pix GAN) │  │
│  │   Conv Net)  │    │               │  │
│  └──────┬───────┘    └───────┬───────┘  │
│         │    Skip Connection  │          │
│         └────────────────────┘          │
│                                         │
│  Loss = L1_SR + L1_Color + GAN +        │
│         Physics-Informed (Emissivity)   │
└─────────┬───────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│   Evaluation        │
│ • PSNR (dB)         │
│ • SSIM              │
│ • FID               │
│ • Visual inspection │
│ • Inference time    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   ONNX Export       │
│ (Deployment-ready)  │
└─────────────────────┘
```

> [!IMPORTANT]
> This flowchart **must** visually match the ISRO Workflow Summary from Slide 6 of their explainer. The judges will immediately recognize that you understood their exact pipeline.

---

### SLIDE 5: Architecture Diagram

**Create a detailed neural network architecture diagram:**

```
INPUT: TIR @ 200m (1×256×256)
         │
    ┌────▼────────────────────────────┐
    │  STAGE 1: Super-Resolution     │
    │  ┌─────────────────────────┐   │
    │  │ Conv2d(1→64, k=5)      │   │
    │  │ PReLU                   │   │
    │  │ Conv2d(64→64, k=3)     │   │
    │  │ PReLU                   │   │
    │  │ SubPixel Conv (2x)     │   │  ──→ OUTPUT 1: TIR @ 100m (1×512×512)
    │  │ Conv2d(64→1, k=3)     │   │
    │  └─────────────────────────┘   │
    └────┬───────────────────────────┘
         │ (64-channel features passed directly)
    ┌────▼────────────────────────────┐
    │  STAGE 2: Colorization (GAN)   │
    │  ┌────────────┐  ┌──────────┐  │
    │  │  Encoder   │  │ Decoder  │  │
    │  │ Conv 64→128│  │DeConv    │  │
    │  │ InstNorm   │  │256→128   │  │
    │  │ LeakyReLU  │→→│InstNorm  │  │
    │  │ Conv128→256│  │ReLU      │  │
    │  │ InstNorm   │  │DeConv    │  │
    │  │ LeakyReLU  │  │128→64   │  │
    │  └────────────┘  └────┬─────┘  │
    │                  + Skip Conn   │  ──→ OUTPUT 2: RGB @ 100m (3×512×512)
    │                  Conv(64→3)    │
    │                  Tanh          │
    └────────────────────────────────┘

    LOSS FUNCTION:
    L_total = λ_SR × L1(SR_out, TIR_GT) 
            + λ_L1 × L1(RGB_out, RGB_GT) 
            + λ_GAN × L_adversarial 
            + λ_physics × L_emissivity
```

**For the pix2pix colorization (standalone training):**
```
Generator: U-Net (8-level encoder-decoder with skip connections)
  - 54M parameters
  - Input: 1-channel TIR → Output: 3-channel RGB

Discriminator: PatchGAN (70×70 receptive field)
  - 2.7M parameters
  - Classifies 30×30 overlapping patches as real/fake
```

---

### SLIDE 6: Technologies to Be Used

**Content to write (in a clean grid/table):**

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Deep Learning** | PyTorch 2.0+ | Model training & inference |
| **Model Architecture** | pix2pix GAN (U-Net + PatchGAN) | Image-to-image translation |
| **Super-Resolution** | Sub-Pixel Convolution (PixelShuffle) | TIR 200m → 100m upscaling |
| **Physics Loss** | Custom Thermal Emissivity Loss | Thermodynamic consistency |
| **Data Handling** | Rasterio, GDAL, tifffile | GeoTIFF satellite data I/O |
| **Image Processing** | OpenCV, scikit-image, Pillow | Preprocessing & augmentation |
| **Evaluation** | PSNR, SSIM (scikit-image), clean-fid | Image quality metrics |
| **Data Acquisition** | Google Earth Engine API | Automated Landsat 9 download |
| **Training Infrastructure** | Google Colab (T4 GPU) | Cloud-based GPU training |
| **Model Export** | ONNX Runtime | Low-latency deployment |
| **Visualization** | React + TypeScript (optional) | Web-based comparison tool |
| **Dataset** | Landsat 9 Level-2 (USGS) | B2, B3, B4, B10 bands |

---

### SLIDE 7: Estimated Implementation Cost

**Content to write:**

| Resource | Cost | Notes |
|----------|------|-------|
| Landsat 9 Data | **Free** | USGS open data, accessed via Google Earth Engine |
| Google Colab (T4 GPU) | **Free** | Free tier sufficient for training (~6-8 hours) |
| Google Colab Pro (optional) | ₹900/month | Faster A100 GPU, longer runtime |
| Python + PyTorch + Libraries | **Free** | All open-source |
| ONNX Runtime | **Free** | Open-source inference engine |
| Cloud Deployment (optional) | ₹0 - ₹500/month | Vercel (free) + Railway (free tier) |
| **Total** | **₹0 — ₹1,400** | Entirely achievable with free/open-source tools |

> The entire solution can be built and deployed at **zero cost** using freely available datasets (USGS Landsat 9), free compute (Google Colab), and open-source frameworks (PyTorch, ONNX). No proprietary tools or paid datasets required.

---

### SLIDE 8 (Optional): Wireframes/Mock Diagrams

**Content:** Include a screenshot of your working web interface showing the before/after slider comparison. This is optional but it shows judges you have a working prototype. Include a 2x2 grid:

1. Upload TIR input image
2. Super-resolved 100m TIR output
3. Colorized RGB output
4. Side-by-side comparison with ground truth

---

## Part 4: Key Phrases to Use in PPT

Use these exact phrases — they directly mirror ISRO's language:

- "End-to-end trained Deep Learning pipeline"
- "Single-channel TIR image @ 200m as input"
- "High-resolution TIR image @ 100m as output"
- "Colorized RGB image @ 100m as output"
- "Physics-informed modeling"
- "Low inference time per tile"
- "Prevent hallucinations through physics constraints"
- "PSNR, SSIM, and FID evaluation"
- "Landsat 9 TIR1-RGB pairs"
- "Bands B2, B3, B4, B10"

---

## Part 5: What NOT to Do in the PPT

> [!CAUTION]
> - ❌ Do NOT say "we already built the full solution." They want an ideation submission, not a demo.
> - ❌ Do NOT mention FLIR dataset. Your old README mentions FLIR ADAS — this is NOT the dataset ISRO wants. They want Landsat 9.
> - ❌ Do NOT over-emphasize the web app. ISRO cares about the ML pipeline, not a React frontend.
> - ❌ Do NOT say you built your own preprocessing. Say you will follow their provided preprocessing workflow (their GitHub repo) and show that you understand the exact steps.
> - ❌ Do NOT make the PPT text-heavy. Use diagrams, flowcharts, and architecture visuals — ISRO explicitly said "drawings/sketches/illustrations add to the power."

---

## Part 6: Immediate Action Items

1. **Update your README.md** — Remove all FLIR dataset references. Make it Landsat 9 only.
2. **Build the PPT** using the slide-by-slide content above.
3. **Create clean diagrams** for the flowchart and architecture slides. Use draw.io, Canva, or PowerPoint SmartArt.
4. **Add FID metric** to your evaluation pipeline (use `clean-fid` library — it's already in your requirements.txt).
5. **Take screenshots** of your preprocessed data samples for the "visual illustrations" slide.
