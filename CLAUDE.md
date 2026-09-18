# CLAUDE.md

Fork of [MedARC-AI/MindEyeV2](https://github.com/MedARC-AI/MindEyeV2) (MindEye2, ICML 2024, arXiv:2403.11207). Upstream reconstructs seen images from fMRI.

**Goal of this fork: build an SAE and other interpretability tooling to characterize the information contained in the representations MindEye2 builds.** Reconstruction benchmarks are not the objective. Treat the trained model as an object of study.

## Ground rules

- **Never read, list, glob, or enumerate `src/nsd_data/`.** ~100 GB, gitignored. Reach it only through the documented `h5py`/`webdataset` paths with explicit filenames.
- **Never train or run inference autonomously.** Write or edit the scripts; I run them. No long jobs, no `accelerate launch`, no notebook execution.
- `src/generative_models/` (Stability SGM) and `src/modeling_git.py` (HuggingFace GIT) are vendored third-party code. Read to understand an API; do not refactor.
- Env is `src/fmri`, managed with uv. Activate it; never run `setup.sh`. Dependency versions are load-bearing — `diffusers.models.vae.Decoder` and the `dalle2_pytorch` internals imported by `models.py` moved in later releases.
- All paths resolve relative to `src/`. Data is `--data_path=./nsd_data`; checkpoints go to `../train_logs/{model_name}` as upstream expects.

## Layout

```
src/
├── Train.ipynb                 training (pretrain + finetune)
├── recon_inference.ipynb       voxels → CLIP embed → recons + captions
├── enhanced_recon_inference.ipynb   SDXL img2img refinement of recons
├── final_evaluations.ipynb     metrics/figures — low priority here
├── dataset_creation.ipynb      how nsd_data was built; reference only, don't rerun
├── models.py                   BrainNetwork, BrainDiffusionPrior, PriorNetwork (+ unused Clipper, eval-only GNet)
├── utils.py                    losses, MixCo, metrics, unclip_recon
├── autoencoder/convnext.py     frozen ConvNeXt-XL teacher for the low-level head
└── *.slurm                     example job scripts
```

Notebooks are the source of truth. They run via `jupyter nbconvert X.ipynb --to python`; the generated `.py` files are build artifacts — edit the `.ipynb`.

## The model

Four stages. Only 1–2 are trained; 3–4 are frozen pretrained generative models.

```
fMRI betas (B, 1, n_vox)                      n_vox is subject-specific
   │
   ▼ [TRAINED] RidgeRegression — one nn.Linear per subject, indexed by subj
shared latent (B, 1, hidden_dim)              the only subject-aligned bottleneck
   │
   ▼ [TRAINED] BrainNetwork — MLP-Mixer + heads
   ├── backbone     (B, 256, 1664)   main CLIP-space output
   ├── clip_voxels  (B, 256, 1664)   backbone → clip_proj; contrastive/retrieval head
   └── blurry       ((B,4,28,28), (B,49,512))   SD-VAE latents + aux maps
   │
   ▼ [TRAINED] BrainDiffusionPrior — conditions on `backbone`, denoises toward true CLIP embed
prior_out (B, 256, 1664)
   │
   ├─▶ [FROZEN] SDXL unCLIP ─────────────▶ 768×768 reconstruction
   └─▶ [FROZEN] CLIPConverter → GIT-large-coco ─▶ predicted caption
```

`clip_seq_dim=256`, `clip_emb_dim=1664` (OpenCLIP ViT-bigG-14, `only_tokens=True`). `hidden_dim` 4096 in the paper, 1024 in the light config. `seq_len=1`, so `mixer_blocks2` acts on a singleton axis. `n_vox` is read at runtime from `betas[0].shape[-1]` — never hardcode it.

`backbone_linear` is `Linear(hidden_dim → 1664*256)`: ~1.7 B params at hidden_dim=4096, ~436 M at 1024. It dominates the model.

Target signal is `clip_target = clip_img_embedder(image)`, `(B, 256, 1664)` — the space every comparison is ultimately against.

## Training flow

WebDataset tars hold **only behavioral arrays**; voxels and images are loaded separately from HDF5 and indexed by behav columns:

- `behav[:,0,0]` = COCO 73k index → `coco_images_224_float16.hdf5`
- `behav[:,0,5]` = global trial → `betas_all_subj0X_fp32_renorm.hdf5` (loaded fully into CPU RAM)

Full column table is in the upstream README. `past_behav` / `future_behav` / `olds_behav` ship in every shard and are **never used by training** — adjacent timepoints, other repetitions of the same image, RT, button press. Free conditioning structure for interpretability.

Per epoch: all batches are bulk pre-loaded into dicts first (much faster than per-batch), with MixCo applied during that pre-load. Then each subject's voxels go through its own ridge index, the latents are concatenated into one batch, and the shared backbone runs once.

```
loss = prior_scale * MSE(prior)                              # default 30
     + clip_scale  * (mixco_nce | soft_clip_loss)            # default 1
     + blur_scale  * (L1(latents) + 0.1 * soft_cont_loss)    # default 0.5
```

Contrastive loss switches from BiMixCo (`mixco_nce`) to SoftCLIP at `mixup_pct` (0.33) through training, with temperature cosine-annealed 0.004 → 0.0075. MixCo mixes **voxels, before the ridge layer**, and the same mixing is applied to the blurry targets.

Two stages: `--multi_subject --subj=N` pretrains on all subjects **except** N; `--no-multi_subject --subj=N` then finetunes on N from that checkpoint via `--multisubject_ckpt`. Exact commands are in the upstream README. Loading a multi-subject checkpoint requires popping `ridge.linears.0.weight` (shapes differ) — `load_ckpt(..., multisubj_loading=True)` does this.

Eval during training averages **3 repetitions** of each test image's voxels, running ridge+backbone per repetition and averaging `clip_voxels` and `backbone`.

## Inference flow

`recon_inference.ipynb` rebuilds the architecture by hand (not imported from Train), loads `last.pth`, and for each unique test image averages 3 voxel repetitions through ridge+backbone. Then:

1. `prior_out = model.diffusion_prior.p_sample_loop(backbone.shape, text_cond=dict(text_embed=backbone), cond_scale=1., timesteps=20)` — DDIM, since 20 < 100 timesteps.
2. Captions: `CLIPConverter` (Linear 256→257 on the sequence axis, 1664→1024 on the embedding axis) → `GitForCausalLMClipEmb.generate(pixel_values=...)`. The GIT patch exists solely so a CLIP embedding can be passed where an image normally goes.
3. Recons: `utils.unclip_recon(...)`, 38 sampler steps.
4. Saves to `evals/{model_name}/*_all_{recons,blurryrecons,predcaptions,clipvoxels}.pt` at 256px.

`enhanced_recon_inference.ipynb` re-noises each recon to `img2img_timepoint=13` and denoises 25 steps with the predicted caption as prompt, CFG 5. This stage **adds generative-prior content** — a confound worth isolating rather than a quality improvement.

## Non-obvious behavior

- **`text_embed` is the brain embedding.** `PriorNetwork.forward` does `brain_embed = text_embed` immediately. No text is involved anywhere in that path.
- Disabled heads return **dummy tensors**, not `None` (`c = Tensor([0.])`). Always check `clip_scale > 0` / `blurry_recon` / `use_prior` before using a return value.
- Training **silently drops** batches containing duplicate images (`if len(image0) != len(image_idx): continue`) — h5py can't fancy-index duplicates.
- `recon_inference.ipynb` deliberately executes a bare `err` when `plotting=True` (auto-set under Jupyter) to halt after one image. Intentional upstream; set `plotting=False` for full runs.
- `ConvnextXL.init_weights` loads with `strict=False` inside a bare `try/except: pass`. Weight-loading failures are silent.
- Inference re-declares config via argparse with **different defaults** than training (`hidden_dim` defaults to 2048). Mismatched `hidden_dim`/`n_blocks`/`blurry_recon` gives shape errors, or silently wrong behavior under `strict=False`.
- Checkpoints may be in DeepSpeed ZeRO format; inference falls back to `zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint`.
- argparse values become bare module-level globals (`for attribute_name in vars(args): globals()[...] = ...`), and `utils.is_interactive()` branches between a hardcoded `jupyter_args` string and real argparse. **Edit both branches** or they silently disagree.

## Interpretability: where to hook

| Site | Shape | Why it matters |
|---|---|---|
| ridge output | `(B, 1, hidden_dim)` | the only subject-aligned bottleneck; most compact; natural first SAE target |
| mixer residual stream | `(B, seq_len, h)` per block | where structure is built; needs hooks, not returned |
| `backbone` | `(B, 256, 1664)` | what the diffusion prior conditions on |
| `clip_voxels` | `(B, 256, 1664)` | contrastively-trained retrieval representation; may diverge from `backbone` |
| `prior_out` | `(B, 256, 1664)` | what unCLIP actually consumes |
| `b_aux` / blurry latents | `(B, 49, 512)` / `(B, 4, 28, 28)` | low-level color/layout, separable from semantics |
| `clip_target` | `(B, 256, 1664)` | ground truth; reference space for all comparisons |

**`backbone` vs `prior_out` is the cleanest handle on brain-derived content vs. generative-prior content** — same space, differing only by the prior's denoising. Both are already logged during training as `recon_cossim` / `recon_mse`.

Guidance:

- **Use forward hooks, not edits to `BrainNetwork.forward`.** `register_forward_hook` on `model.ridge`, `model.backbone.mixer_blocks1[i]`, `backbone_linear`, `clip_proj` gets everything while keeping checkpoints loading with `strict=True`. If `forward` must change, add an opt-in `return_intermediates=False` and never alter the default three-value return — all three notebooks unpack exactly three.
- Put new code in **`src/interp/`**. Keeping upstream files near-pristine makes rebasing possible and makes it clear what is original work.
- Activations are fp16 under autocast. Cast to fp32 before accumulating statistics or fitting an SAE.
- The pipeline is deterministic given fixed seeds **except** prior sampling and unCLIP, which draw noise. `p_sample` takes a `generator` but currently ignores it for the noise draw (see the commented line) — fix that if exact reproducibility is needed.
- Start activation collection on the test set (3000 items for most subjects) rather than the 40-session train set.
- For subject-generality questions, the ridge layer is the lever: everything from its output onward is shared across subjects, so anything subject-specific must live in `ridge.linears[i]`.
