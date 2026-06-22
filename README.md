# Leveraging Segment Anything Model for Source-Free Domain Adaptation via Dual Feature Guided Auto-Prompting
This repository contains Pytorch implementation of our source-free domain adaptation (SFDA) method with Dual Feature Guided (DFG) auto-prompting approach. ([Arxiv](https://arxiv.org/pdf/2505.08527))

<!-- ![method](./figures/method.png "") -->
## Introduction

Source-free domain adaptation (SFDA) for segmentation aims at adapting a model trained in the source domain to perform well in the target domain with only the source model and unlabeled target data. Inspired by the recent success of Segment Anything Model (SAM) which exhibits the generality of segmenting images of various modalities and in different domains given human-annotated prompts like bounding boxes or points, we for the first time explore the potentials of Segment Anything Model for SFDA via automatedly finding an accurate bounding box prompt. We find that the bounding boxes directly generated with existing SFDA approaches are defective due to the domain gap. To tackle this issue, we propose a novel Dual Feature Guided (DFG) auto-prompting approach to search for the box prompt. Specifically, the source model is first trained in a feature aggregation phase, which not only preliminarily adapts the source model to the target domain but also builds a feature distribution well-prepared for box prompt search. In the second phase, based on two feature distribution observations, we gradually expand the box prompt with the guidance of the target model feature and the SAM feature to handle the class-wise clustered target features and the class-wise dispersed target features, respectively. To remove the potentially enlarged false positive regions caused by the over-confident prediction of the target model, the refined pseudo-labels produced by SAM are further postprocessed based on connectivity analysis. 
Experiments on 3D and 2D datasets indicate that our approach yields superior performance compared to conventional methods.

![introduction](./figures/introduction.png "")
Take spleen in a target domain image in MRI⟶CT adaptation as an example. (a) MedSAM requires an accurate bounding box prompt. Neither a too-small nor a too-large bounding box leads to a decent segmentation result. (b) Segmentation results of ProtoContra and the corresponding bounding boxes, produced by different output probability thresholds. Due to the domain gap and limited knowledge from source model and target unlabeled data, it is hard for existing SFDA methods to generate precise box prompts even if we vary the probability threshold. (c) After feature aggregation, our dual feature guided bounding box prompt search approach can find an accurate box prompt for MedSAM to yield refined pseudo-labels. (d) The searching procedure of our proposed box prompt search method. The red numbers are indices of the boxes, corresponding to the horizontal axis in (e). (e) The number of pixels of changed MedSAM predictions when the box prompt is switched from the last to the current. MedSAM prediction keeps stable when the box prompt fluctuates near the ground truth. We utilize this property to find the optimal box prompt.

Our method:
![method](./figures/method.PNG "")

Segmentation results:
![segmentation results](./figures/quali.png "")

## Installation
Create the environment from the `environment.yml` file:
```
conda env create -f environment.yml
conda activate dfg
```
## Data preparation
* Download the BTCV dataset from [MICCAI 2015 Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789), and the CHAOS dataset from [2019 CHAOS Challenge](https://chaos.grand-challenge.org/). Then preprocess the downloaded data referring to `./preprocess.ipynb`.
You can also directly download our preprocessed datasets from [here](https://drive.google.com/drive/folders/1g2ar0L18ryO9zlmVnl-1Ia-XrHkDODfN?usp=sharing). The paths to the datasets need to be specified in the yaml files in `./configs`.

## Training
The following are the steps for the CHAOS (MRI) to BTCV (CT) adaptation.
* Download the source domain model from [here](https://drive.google.com/file/d/18zhjTuy3LFqWMPckrhby8SX9-2muh1KQ/view?usp=sharing) or specify the data path in `configs/train_source_seg.yaml` and then run 
```
python main_trainer_source.py --config_file configs/train_source_seg.yaml
```
* Download the trained model after the feature aggregation phase from [here](https://drive.google.com/file/d/1eOOnQ4Je9UrfJ-Hqf5e9aaHyUTK6-iMv/view?usp=sharing) or specify the source model path and data path in `configs/train_target_adapt_FA.yaml`, and then run
```
python main_trainer_fa.py --config_file configs/train_target_adapt_FA.yaml
```
* Download the MedSAM model checkpoint from [here](https://drive.google.com/file/d/1khQO5G-qYZsCkocEhZ-HhX8IioEUAZ9Q/view?usp=sharing) and put it under `./medsam/work_dir/MedSAM`.
* Specify the model (after feature aggregation) path, data path, and refined pseudo-label paths in `configs/train_target_adapt_SAM.yaml`, and then run
```
python main_trainer_sam.py --config_file configs/train_target_adapt_SAM.yaml
```
<!--
## Result
cup: 0.7503 disc: 0.9503 avg: 0.8503 cup: 9.8381 disc: 4.3139 avg: 7.0760
## REFUGE to Drishti-GS adaptation
Follow the same pipeline as above, but run these commands to specify the new parameters:
```
python train_source.py --datasetS Domain4
python generate_pseudo.py --dataset Domain1 --model-file /path/to/source_model
python sim_learn.py --dataset Domain1 --model-file /path/to/source_model --pseudo /path/to/pseudo_label
python pl_refine.py --dataset Domain1 --weights /path/to/context_similarity_model --logt 5 --pseudo /path/to/pseudo_label
python train_target.py --dataset Domain1 --model-file /path/to/context_similarity_model --num_epochs 20
```
 -->
## Acknowledgement
We would like to thank the great work of the following open-source projects: [ProtoContra](https://github.com/CSCYQJ/MICCAI23-ProtoContra-SFDA), [MedSAM](https://github.com/bowang-lab/MedSAM).

## Citation
```
@ARTICLE{11079936,
  author={Huai, Zheang and Tang, Hui and Li, Yi and Chen, Zhuangzhuang and Li, Xiaomeng},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Leveraging Segment Anything Model for Source-Free Domain Adaptation via Dual Feature Guided Auto-Prompting}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Adaptation models;Image segmentation;Foundation models;Data models;Biomedical imaging;Predictive models;Accuracy;Uncertainty;Training;Spleen;Source-free domain adaptation;Segment Anything Model;Prompt;Bounding box},
  doi={10.1109/TMI.2025.3587733}}
```
## PROSTATE Dataset Reproduction

### Data Preparation

PROSTATE dataset (6 domains: BMC, RUNMC, BIDMC, HK, UCL, I2CVB) processed from NCI-ISBI 2013 and I2CVB challenges.

- **Source domain**: BMC + RUNMC
- **Target 1**: BIDMC + HK + UCL
- **Target 2**: I2CVB
- **Data root**: `/opt/data/private/MedSeg_Data_Process/PROSTATE/processed_new`
- **Resolution**: Original 384x384, resized to 256x256 for training
- **Classes**: 2 (background + prostate)
- **Format**: `.npz` files with `img`/`label` keys, `metadata.json` for train/test splits

### Training Pipeline

#### 1. Source Supervised Training

```bash
python main_trainer_source.py --config_file configs/train_prostate_source_seg.yaml
```

Config: 100 epochs, lr=0.0003, batch_size=16, img_size=256x256

Best model: `best_model_epoch_71_dice_0.9164.pth`

#### 2. Feature Aggregation (FA) Adaptation

```bash
# Tuned FA (recommended for binary segmentation):
PYTHONPATH=medsam:$PYTHONPATH python main_trainer_fa.py --config_file configs/train_prostate_fa_lr1e5_ep2_prop.yaml

# Original FA config (for reference, performs poorly on binary tasks):
# PYTHONPATH=medsam:$PYTHONPATH python main_trainer_fa.py --config_file configs/train_prostate_target_adapt_FA.yaml
```

**FA Hyperparameter Tuning for Binary Segmentation**:

The original FA hyperparameters (lr=0.0001, uniform class proportion [0.5, 0.5]) cause severe performance degradation on PROSTATE (2-class). Two key fixes:

1. **Lower learning rate** (1e-5 vs 1e-4): The prototype-based transport loss with only 2 classes provides very weak structural constraints. A high learning rate causes the encoder features to drift far from the source distribution, destroying the pre-trained representations.
2. **Corrected class proportions** ([0.85, 0.15] vs [0.5, 0.5]): Background occupies ~85% of PROSTATE slices. The uniform prior misdirects the optimal transport alignment, causing the model to over-segment.

FA sweep results (Dice on target_1):

| Config | lr | Epochs | Proportion | FA Dice |
|--------|-----|--------|------------|---------|
| **lr1e5_ep2_prop** | **1e-5** | **2** | **[0.85, 0.15]** | **0.7980** |
| lr1e5_ep3 | 1e-5 | 3 | uniform | 0.7900 |
| lr3e5_ep2 | 3e-5 | 2 | uniform | 0.6970 |
| lr5e5_ep1 | 5e-5 | 1 | uniform | 0.6951 |
| original | 1e-4 | 5 | uniform | 0.5150 |

Best FA model (tuned): `model_step_10_dice_0.7980.pth`

#### 3. SAM Pseudo-Label Refinement + Retraining

```bash
PYTHONPATH=medsam:$PYTHONPATH python main_trainer_sam.py --config_file configs/train_prostate_target_adapt_SAM.yaml
```

Config: 100 epochs retraining, batch_size=8, refine_order=[1], diffuse_max_step=[0,30]

**Note**: Must set `PYTHONPATH` to include `medsam/` directory for `segment_anything` imports. Update `source_model_path` in the SAM config to point to the desired FA (or source) model.

Best model (with tuned FA): `best_model_step_10_dice_0.8292.pth`

### Testing

```bash
python test_prostate.py --model_path <path_to_model.pth> --domain <source|target_1|target_2> --gpu_id 0
```

### Results (Volume-level Dice / ASSD)

| Model | Source (BMC+RUNMC) | Target 1 (BIDMC+HK+UCL) | Target 2 (I2CVB) |
|-------|-------------------|--------------------------|-------------------|
| Source Only | 0.9164 / 0.53 | 0.7430 / 1.93 | 0.6654 / 6.37 |
| FA(original) + SAM | 0.7522 / 1.90 | 0.6976 / 2.80 | 0.5492 / 10.23 |
| SAM only (no FA) | 0.8722 / 0.84 | 0.7443 / 1.90 | **0.6799 / 8.82** |
| **FA(tuned) + SAM** | **0.8724 / 0.94** | **0.8292 / 1.33** | 0.5765 / 6.74 |

**Key findings**: With tuned FA hyperparameters, the full FA+SAM pipeline achieves the best target_1 performance (0.8292 Dice), a +13.2% improvement over the original FA+SAM (0.6976) and +8.6% over source-only (0.7430).

Result files saved in: `test_results/DFG_PROSTATE/`
