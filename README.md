# LCA-on-the-Line : Benchmarking Out of Distribution Generalization with Class Taxonomies (*ICML 2024 Oral Presentation*)

## [Project Page](https://elvishelvis.github.io/papers/lca/)


<!-- ## Quick Start

```bash
conda env create -f environment.yml
conda activate clip
``` -->


## Experiment

### Correlation Experiment

1). Pre-extract model logits using the following scripts:

- [extract_clip_feature.py](./prepare_logit/extract_clip_feature.py)

- [extract_openclip_feature.py](./prepare_logit/extract_openclip_feature.py)

- [prepare_model_candidates.py](./prepare_logit/prepare_model_candidates.py)

Alternatively, download pre-extracted logits from [this link](https://huggingface.co/datasets/elvishelvis6/LCA-on-the-Line).

2). Modify the `logit_folder_path` in `main.py` to the path where the logits are stored.

3). Launch the experiment:

```
python main.py
```

### Soft Labels Linear Probing Experiment

1). Pre-extract backbone model features using  [extract_feature_linear_probe.py](./prepare_logit/extract_feature_linear_probe.py). Alternatively, download pre-extracted backbone features from [this link](https://huggingface.co/datasets/elvishelvis6/LCA-on-the-Line/tree/main/linear_feature).

2). Modify `DATASET_PATHS` in `linear_probe_runner.py` accordingly with the output path from the previous step.

3). Launch experiment.

```
python linear_probe_runner.py
```

### Latent hierarchy construction

Follow the instructions in `create_hierarchy.py` (Models' logits are required).

To use the latent hierarchy in previous experiments, follow the `use_latent_hierarchy` section in `main.py`, and update `tree_list` in `linear_probe_runner.py`.

To evaluate the correlation between ID LCA and Soft Labels quality, refer to the "Predict soft label quality with source model LCA that construct latent hierarchy" section in `main.py`.

## Thanks!

The scripts `hier.py`, `datasets/`, and `wordNet_tree.npy` are adopted from [Jack Valmadre's hierarchical classification repository](https://github.com/jvlmdr/hiercls) 
- Valmadre, Jack. "Hierarchical classification at multiple operating points." Advances in Neural Information Processing Systems 35 (2022): 18034-18045.


## To-Do
- [ ] Add environment.
- [ ] Clean up code for readibility.
- [ ] Make variables/paths configurable as flags.


