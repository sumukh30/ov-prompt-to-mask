# 🛸 Real-Time Open-Vocabulary Prompt-to-Mask Video Analytics

> 📂 Full project (datasets, checkpoints, outputs): [Google Drive](https://drive.google.com/drive/folders/1pymvoVwySclelGfF5iGmglZwzj0jel8X?usp=sharing)

---

## 🧠 What This Project Does

Type a text prompt like `"car, pedestrian, bus"` — point it at a drone video — and get back pixel-level segmentation masks tracked across every frame. No fixed class list. No retraining needed between queries.

The pipeline chains two foundation models:

- 🔍 **YOLO-World** — detects objects on frame 0 using your text prompt via CLIP alignment
- 🎭 **SAM-2** — turns each detection box into a pixel mask, then tracks it across all remaining frames automatically

Fine-tuning YOLO-World on VisDrone improved mAP@0.5 by **+177%** and made three previously undetectable aerial classes detectable. An ablation study showed **69% of that gain** came from weight-level domain adaptation — not prompt engineering.

---

## 🚀 Try the Demo

Open the self-contained demo notebook in Colab. Runs on **free T4 GPU** in ~8 minutes with no Drive mount or setup required.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tvwZq_ZlY-AJWpgSkvhv8LO-f_Ky54Zd)


---

## 📊 Results at a Glance

| Metric | Pretrained | Fine-tuned |
|---|---|---|
| mAP@0.5 (VisDrone val) | 0.1036 | **0.2874** (+177%) |
| mAP@0.5 (stress set) | 0.0882 | **0.2900** (+229%) |
| mAP-small (< 32×32 px) | 0.0504 | **0.1606** (+219%) |
| YOLO-World FPS | 34.3 | **36.0** ✅ real-time |
| SAM-2 mean mask IoU | — | **0.8085** |
| Video tracking coverage | — | **100%** (6/6 videos) |

---

## 🏗️ Pipeline Architecture

```
Text Prompt ──[CLIP]──▶ Semantic Embeddings
                                │
Drone Video ──Frame 0──▶ YOLO-World ──▶ Bounding Boxes + Class Labels
                                │
                      SAM-2 ImagePredictor ──▶ Pixel Masks (single image)

                      SAM-2 VideoPredictor
                      ┌─────────────────────────────┐
                      │  init_state(frame_folder/)  │
                      │  add_new_points_or_box() ×N │
                      │  propagate_in_video()        │
                      └─────────────────────────────┘
                                │
                      Tracked masks across all frames
```

---

## 📁 Repository Structure

```
📦 root
│
├── 📓 demo_pipeline.ipynb            ← Self-contained demo (start here)
│
├── 📂 notebooks/
│   ├── 📂 baseline/                  ← Baseline evaluation notebooks
│   ├── 📂 env_setup/                 ← Environment setup & Drive config
│   ├── 📂 final pipeline/            ← End-to-end demo + MP4 export
│   ├── 📂 fine-tuning/               ← 50-epoch YOLO-World fine-tuning notebooks
│   ├── 📂 NLP eval/                  ← Open-vocabulary NLP prompt experiments
│   ├── 📂 preprocessing/             ← VisDrone / COCO / YouTube-VIS preprocessing
│   ├── 📂 UI/                        ← Streamlit web app notebooks
│   └── 📂 video-tracking/            ← SAM-2 VideoPredictor tracking pipeline
│
├── 📂 dataset/
│   ├── raw/                          ← Raw dataset downloads (see Drive link above)
│   └── processed/                    ← YOLO-format labels & splits (see Drive link above)
│
└── 📄 README.md
```

> 📌 **Note:** `dataset/raw/` and `dataset/processed/` are empty placeholder folders in this repo.  
> The full preprocessed datasets (~15 GB total) are on [Google Drive](https://drive.google.com/drive/folders/1pymvoVwySclelGfF5iGmglZwzj0jel8X).

---

## 🗄️ Datasets Used

| Dataset | Purpose | Scale |
|---|---|---|
| [VisDrone-DET](https://github.com/VisDrone/VisDrone-Dataset) | Primary detection benchmark — 11 aerial classes | 6,471 train / 548 val images |
| [COCO 2017](https://cocodataset.org) | SAM-2 segmentation quality evaluation | 100 val images, 6 overlapping classes |
| [YouTube-VIS 2021](https://youtube-vos.org/dataset/vis/) | Video tracking evaluation | 2,985 videos / 90,160 extracted frames |

---

## 🔧 Models & Key Hyperparameters

### YOLO-World Fine-Tuning

| Parameter | Value | Reason |
|---|---|---|
| Base checkpoint | `yolov8l-world.pt` | Largest YOLO-World variant |
| Epochs | 50 (best: ep. 49) | Model still improving at final epoch |
| Optimizer | AdamW | Stable convergence for fine-tuning |
| Learning rate (`lr0`) | **0.001** (not 0.01) | Prevents overwriting pretrained CLIP embeddings |
| Mosaic augmentation | 1.0 → off last 10 epochs | Increases rare-class exposure during training |
| Vertical flip | 0.5 | Valid for aerial imagery |
| Rotation | 0.0 (disabled) | Drones maintain level attitude |
| Hardware | NVIDIA A100 40 GB | ~99 minutes total |

### SAM-2 (Used Pretrained — No Fine-Tuning)

| Setting | Value |
|---|---|
| Model | SAM-2.1 Hiera Large |
| Image mode | `SAM2ImagePredictor`, `multimask_output=False` |
| Video mode | `SAM2VideoPredictor`, box prompts on frame 0 |
| Config path | `configs/sam2.1/sam2.1_hiera_l.yaml` *(must be relative — Hydra requirement)* |

---

## 🧪 Reproducing the Experiments

All notebooks run on Google Colab with Drive mount. To reproduce from scratch:

**1. Clone this repo**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

**2. Download data & checkpoints from Drive**  
👉 [Google Drive](https://drive.google.com/drive/folders/1pymvoVwySclelGfF5iGmglZwzj0jel8X?usp=sharing)  
Place files under `dataset/raw/` and `dataset/processed/` matching the paths in `configs/config.json`.

**3. Run notebooks in order**
```
0_setup
  → 1_dataset_audit
  → 2_preprocess_visdrone + 3_preprocess_coco + 4_preprocess_ytvis
  → 2_baseline_evaluation
  → 03_finetune_yoloworld
  → 04_video_tracking
  → 5_final_pipeline
```

**4. Or just run the demo**  
Open `demo_pipeline.ipynb` in Colab — it downloads everything it needs automatically.

---

## 🛠️ How It Was Built

The project was built end to end across data, training, and infrastructure.

**Data pipelines.** The preprocessing work covers three datasets with distinct annotation formats. For VisDrone, we wrote a converter handling edge cases the raw annotations don't document — ignored regions (category 0 in VisDrone means "ignore", not a class), degenerate boxes with zero or negative dimensions, coordinates that run slightly outside the image frame, and one-based category indexing that differs from YOLO's zero-based scheme. For COCO, category IDs are non-sequential (1, 2, 3, 4, 6, 8...) so we remapped them to clean 0-based YOLO IDs and filtered from 80 classes down to the 6 that overlap with VisDrone. For YouTube-VIS, annotations are structured per-video with per-frame bbox lists where `None` means the object is absent that frame — careful `None` handling was needed to avoid writing malformed label files. A class imbalance analysis across all 6,471 training images revealed that car dominates at 42% of boxes while awning-tricycle sits under 3%, and this directly informed the augmentation strategy.

**Fine-tuning.** We used a deliberately low learning rate (0.001 vs the default 0.01) to prevent catastrophic forgetting of the pretrained CLIP alignment embeddings. Those embeddings are what enable open-vocabulary detection — overwriting them with aggressive gradient updates improves VisDrone accuracy while breaking generalization to novel text prompts. Mosaic augmentation was set to maximum probability for the first 40 epochs to increase rare-class exposure in each batch, then disabled for the final 10 epochs to let the model stabilize on clean single-image samples. A checkpoint backup callback saved `last.pt` after every epoch so training could resume from the last completed epoch if the 99-minute Colab session disconnected.

**Ablation study.** Rather than simply reporting the mAP improvement, we isolated whether the gain came from weight adaptation or better prompts by running three conditions on the same 548 validation images: pretrained + generic prompts (mAP 0.028), pretrained + VisDrone-specific prompts (mAP 0.107), fine-tuned (mAP 0.287). The 69/31 split between weight adaptation and prompt engineering is the core finding — it shows that for aerial detection specifically, writing better class names is not a substitute for training on aerial data. The motor and people classes scored 0.000 in both pretrained conditions regardless of the prompt, confirming that no text description can make the model detect objects it has no learned visual representation for.

**Pipeline integration.** Connecting YOLO-World to SAM-2 required fixing two non-obvious shape issues. SAM-2 returns mask logits with shape `(N, 1, H, W)` — the extra dimension is an API detail supporting multi-mask output — and using these directly as boolean indices throws an `IndexError`. Squeezing dimension 1 before use fixes it. For video tracking, `reset_state()` must be called after every video to release the frame cache SAM-2 accumulates in GPU memory; without this the session crashes after the second or third clip. The full end-to-end pipeline — loading the fine-tuned checkpoint, running VideoPredictor across multi-frame sequences, rendering coloured mask overlays per tracked object, and exporting annotated MP4s — was assembled in `5_final_pipeline.ipynb`. The Streamlit app accepts comma-separated text prompts, runs YOLO-World with `set_classes()`, optionally passes detections to SAM-2 for pixel masks, and returns annotated results with detection counts and latency.

**Infrastructure fixes.** Two failures were diagnosed during the project. The Google Drive FUSE mount silently loses data: Python's file I/O appears to succeed because writes go to a kernel buffer, but that buffer never flushes to Drive storage before the Colab runtime disconnects — we lost a complete 10 GB preprocessing run this way. The fix is writing to Colab's local SSD first, then transferring via `tar && cp && sync` to force the kernel to commit all pending writes before returning. The SAM-2 hydra import failure (`No module named sam2.build_sam`) looked like a missing package but was a timing issue: `sam2/__init__.py` calls `hydra.initialize_config_module()` at import time, and hydra installed in one cell is only visible to the interpreter from the next cell onward. Splitting install and import into two separate cells resolved it.

---

## 💡 Engineering Notes

| Problem | Root Cause | Fix |
|---|---|---|
| 🔴 **Drive data disappeared after session** | FUSE mount buffers writes in kernel memory — never flushes before runtime disconnects | Write to local SSD, then `tar` + `cp && sync` to Drive |
| 🔴 **`No module named sam2.build_sam`** | `sam2/__init__.py` imports hydra at load time — hydra not yet visible in the same cell it was installed | Split install and import into two separate cells |
| 🔴 **`MissingConfigException` in SAM-2** | Absolute path passed to `build_sam2()` — Hydra requires package-relative paths | Pass `"configs/sam2.1/sam2.1_hiera_l.yaml"` with `cwd=/content/sam2` |
| 🔴 **`IndexError` on SAM-2 masks** | SAM-2 returns `(N, 1, H, W)` logits — extra dim breaks boolean indexing | Squeeze: `masks[:, 0, :, :]` → `(N, H, W)` |
| 🔴 **VRAM OOM after 2nd video** | SAM-2 VideoPredictor accumulates a frame cache across videos | Call `reset_state()` + `torch.cuda.empty_cache()` after every video |

---

## 🖥️ Streamlit App

The project includes a browser-based interactive demo.

**To run on Colab:**
1. Open `UI.ipynb`
2. Run all cells — a public URL is printed via ngrok
3. Open the URL, enter a comma-separated prompt (e.g. `"car, pedestrian"`), upload a drone image, click **Detect**

**Sidebar controls:**
- 🎚️ Confidence threshold (0.1 – 0.9)
- 🎭 SAM-2 segmentation toggle
- 🔀 Fine-tuned vs. pretrained model selector

---

## 🔭 Future Work

- **⚡ Speed up the pipeline** — Replace SAM-2 Large with SAM-2 Tiny to substantially improve throughput. The full pipeline currently runs at ~0.3 FPS (3.5 s/frame on T4) due to SAM-2's propagation cost, which rules out live surveillance. SAM-2 Tiny would cut this significantly at modest quality cost.

- **🔄 Keyframe re-detection** — Currently YOLO-World only runs on frame 0. Adding re-detection every N frames would handle objects that enter or exit the scene mid-video, which the current pipeline misses entirely.

- **🗂️ Joint VisDrone + COCO training** — COCO data for the 6 overlapping classes (person, bicycle, car, motorcycle, bus, truck) was fully preprocessed but not used during fine-tuning. Training jointly on both datasets would test whether ground-level examples of overlapping categories improve aerial detection further.

- **✂️ Domain-specific SAM-2 adaptation** — VisDrone has no pixel-level segmentation ground truth, so SAM-2 was used pretrained. Annotating even a small subset of VisDrone images with masks and fine-tuning SAM-2 on them could improve segmentation quality on tiny aerial objects.

- **🌐 Broader open-vocabulary testing** — The current NLP demo showed CLIP alignment works well for concrete object nouns but fails on action-based prompts ("object in motion"). Training with region-level text supervision could extend the vocabulary to cover attributes and actions.

---

## 📜 References

1. P. Zhu et al., "Detection and Tracking Meet Drones Challenge," *IEEE TPAMI*, 2021.
2. T. Cheng et al., "YOLO-World: Real-Time Open-Vocabulary Object Detection," *CVPR 2024*.
3. N. Ravi et al., "SAM 2: Segment Anything in Images and Videos," *arXiv:2408.00714*, 2024.
4. A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," *ICML 2021*.
5. L. Yang et al., "Video Instance Segmentation," *ICCV 2019*.

---

<p align="center">
  Built with &nbsp;
  <a href="https://github.com/AILab-CVC/YOLO-World">YOLO-World</a> &nbsp;·&nbsp;
  <a href="https://github.com/facebookresearch/sam2">SAM-2</a> &nbsp;·&nbsp;
  <a href="https://github.com/VisDrone/VisDrone-Dataset">VisDrone</a>
</p>