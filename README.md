<h1 align="center">Aligning Multimodal Representations through an Information Bottleneck</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2506.04870"><img src="https://img.shields.io/badge/arXiv-2506.04870-b31b1b.svg"></a>
  <img src="https://img.shields.io/badge/Python-3.8-blue.svg">
  <img src="https://img.shields.io/badge/PyTorch-1.12.1-green.svg">
</p>

<p align="center">
  <i>Official implementation of <b>Aligning Multimodal Representations through an Information Bottleneck</b> (ICML 2025)</i><br>
  <b>Antonio Almudévar</b>, José Miguel Hernández-Lobato, Sameer Khurana, Ricard Marxer, Alfonso Ortega
</p>


## 🌟 Overview
Contrastive losses maximize shared information but retain modality-specific content, causing misalignment. We formalize this with information theory and propose a lightweight regularizer to improve cross-modal representation alignment.

<p align="center">
  <img src="docs/abstract.svg" width="550" alt="Method overview">
</p>


## 📜 Abstract
Contrastive losses have been extensively used as a tool for multimodal representation learning. However, it has been empirically observed that their use is not effective to learn an aligned representation space. In this paper, we argue that this phenomenon is caused by the presence of modality-specific information in the representation space. Although some of the most widely used contrastive losses maximize the mutual information between representations of both modalities, they are not designed to remove the modality-specific information. We give a theoretical description of this problem through the lens of the Information Bottleneck Principle. We also empirically analyze how different hyperparameters affect the emergence of this phenomenon in a controlled experimental setup. Finally, we propose a regularization term in the loss function that is derived by means of a variational approximation and aims to increase the representational alignment. We analyze in a set of controlled experiments and real-world applications the advantages of including this regularization term.




## 🚀 Installation
```
git clone https://github.com/antonioalmudevar/multimodal_ib.git
cd multimodal_ib
python3 -m venv venv
source ./venv/bin/activate
python setup.py develop
pip install -r requirements.txt
```


## 🌀 Disentanglement Results

#### Does the contrastive loss alone remove nuisances?
To test whether **contrastive loss** by itself is sufficient to remove nuisance information, run:
```
python remove_msi.py <config_file> <missing_factor> --seed <seed_value>
```
The <config_file> specifies default hyperparameters such as the temperature and the image encoder.
To override these for sensitivity analysis, use:
```
python remove_msi.py <config_file> <missing_factor> \
    --seed <seed_value> \
    --temperature <temperature_value> \
    --img_encoder <img_encoder_value>
```

#### Does the presence of nuisances in the representation negatively correlate with the level of alignment?
To run a single trial of this correlation analysis:
```
python run_alignment.py <config_file> --seed <seed_value>
```

#### Does our regularization term effectively increase the alignment level?
To assess the effect of the regularization term on alignment accuracy, run:
```
python run_regularization.py <config_file> --beta <beta_value> --seed <seed_value>
```
Replace <beta_value> with the desired regularization strength.

## 📝 Captioning Results
To train the model with our loss on real data and measure its effect on text generation and retrieval performance:
```
python train_captioning.py <config_file> --seed <seed_value>
```


## 📂 Repository Structure
```
.
├── bin/                  # Reproduction scripts
├── configs/              # YAML configs for experiments
├── docs/                 # Figures and docs
├── results/              # Saved outputs
└── src/                  # Core model and training code
```

## 📚 Citation
```
@inproceedings{almudevar2025aligning,
    title={Aligning Multimodal Representations through an Information Bottleneck},
    author={Almud{\'e}var, Antonio and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel and Khurana, Sameer and Marxer, Ricard and Ortega, Alfonso},
    booktitle={International Conference on Machine Learning},
    year={2025}
}
```
