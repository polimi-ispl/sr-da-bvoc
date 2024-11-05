# :earth_africa: :seedling: sr-da-bvoc  :seedling: :earth_africa:
--

This repository contains the code implementation for the paper:

Giganti, A.; Mandelli, S.; Bestagini, P.; Tubaro, S. [*"Learn from Simulations, Adapt to Observations: Super-Resolution of Isoprene Emissions via Unpaired Domain Adaptation"*](https://www.mdpi.com/2072-4292/16/21/3963). Remote Sens. 2024, 16, 3963. https://doi.org/10.3390/rs16213963

[![Remote Sensing](https://img.shields.io/badge/MDPI-2024-green.svg?style=flat-square)](https://www.mdpi.com/2072-4292/16/21/3963)

##Table of Contents 

- [Overview](##Overview)
- [Installation](##Installation)
- [Requirements](##Requirements)
- [Usage](##Usage)
- [Project Structure](##ProjectStructure)
- [Biogenic Inventories](##BiogenicInventories)
- [Citation](##Citation)
- [License](##License)
- [Acknowledgement](##Acknowledgement)
- [Misc](##Misc)

##Overview
### Abstract
*Plants emit Biogenic Volatile Organic Compounds (BVOCs), such as isoprene, significantly influencing atmospheric chemistry and climate. BVOC emissions estimated from bottom-up (BU) approaches (derived from numerical simulations) usually exhibit denser and more detailed spatial information concerning the ones estimated through top-down (TD) approaches (derived from satellite observations). Moreover, numerically simulated emissions are typically easier to obtain, even if they are less reliable than satellite acquisitions which, being derived from actual measurements, are considered a more trustworthy instrument to perform chemistry and climate investigations. Given the coarseness and relative lack of satellite-derived measurements, fine-grained numerically simulated emissions could be exploited to enhance them. However, simulated (BU) and observed (TD) emissions usually differ regarding value range and spatiotemporal resolution. In this work, we present a novel Deep Learning (DL)-based approach to increase the spatial resolution of satellite-derived isoprene emissions, investigating the adoption of efficient Domain Adaptation (DA) techniques to bridge the gap between numerically simulated emissions and satellite-derived ones, avoiding the need of retraining a specific Super-Resolution (SR) algorithm on them. To do so, we propose a methodology based on the Cycle Generative Adversarial Network (CycleGAN) architecture, which has been extensively used for adapting natural images (like digital photographs) of different domains. In our work, we depart from the standard CycleGAN framework, proposing additional loss terms that allow for better DA and emissions’ SR. We extensively demonstrate the proposed method’s effectiveness and robustness in restoring fine-grained patterns of observed isoprene emissions. Moreover, we compare different setups and validate our approach using different emission inventories from both domains. Eventually, we show that the proposed DA strategy paves the way towards robust SR solutions even in case of spatial resolution mismatch between the training and testing domains and in case of unknown testing data.*

### System
![deployment](https://www.mdpi.com/remotesensing/remotesensing-16-03963/article_deploy/html/images/remotesensing-16-03963-g003.png)

##Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/polimi-ispl/sr-da-bvoc.git
```

##Requirements

All the required packages are contained in the ```bvoc-env.yml``` file.

You can install all the packages using [Conda](https://docs.conda.io/projects/conda/en/latest/index.html), by executing the following commands:
```bash
conda env create -f bvoc-env.yml
conda activate bvoc-env
```

More info [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

##Project Structure
```bash
<usr_dir>
├── sr								# Super-Resolution folder
│  	 ├── metrics/
│	 ├── models/
│	 ├── dataset.py
│	 ├── train.py
│	 └── test.py
├── da								# Domain Adaptation folder
│	 ├── models.py
│	 ├── utils.py
│	 └── data_utils.py
├── runs/ 							# Runs folder
│	 ├── end2end
│	 │	    └── logs/				# Tensorboard logs folder
│	 └── sr				
├── utils.py
├── dataset.py
├── train.py
├── test.py
├── bvoc-env.yaml					# Conda environment settings
├── bash.sh					
└── README.md						# This file ;)
```   

##Usage

To train the models, you can execute the following command:
```bash
python train.py --source_domain_dataset <source_domain_dataset> --target_domain_dataset <target_domain_dataset> --em_consistency_loss --feature_alignment_loss --gamma 0.5 --delta 0.0 --ptr_sr --ptr_sr_run_flag <pretrained_sr_net_run_flag>
```

To test the models, you can execute the following command:
```bash
python test.py --train_run_timestamp <train_run_timestamp> --main_dataset <lr_dataset> --source_domain_dataset <source_domain_dataset> --target_domain_dataset <target_domain_dataset>  --ptr_sr --ptr_sr_run_flag <pretrained_sr_net_run_flag>
```

Refer to the ```train.py``` and ```test.py``` files for a more detailed explanation. 

##Biogenic Inventories
The train and test emission are pairs of high resolution (HR) and low resolution (LR) emission map patches.
The dataset folder structure is:

```bash
<dataset_dir>
├── HR/							  	# high-resolution maps
│   ├── train/
│   │   ├── map1
│   │   ├── ...
│   ├── val/
│   └── test/
└── LR/							 	# low-resolution maps
    ├── train/
    │   ├── map1
    │   ├── ...
    ├── val/
    └── test/
``` 

- **TD-OMI-050** and **TD-GOME2-050** emission inventories are available at [https://emissions.aeronomie.be/](https://emissions.aeronomie.be/)
- **BU-MEG-025** emission inventory is available at [https://permalink.aeris-data.fr/CAMS-GLOB-BIO](https://permalink.aeris-data.fr/CAMS-GLOB-BIO)
- **BU-MEG-050** emission inventory is available at [https://doi.org/10.57760/sciencedb.iap.00008](https://doi.org/10.57760/sciencedb.iap.00008)
 
##Citation

```bibtex
@article{giganti2024bvoc,
author = {Giganti, Antonio and Mandelli, Sara and Bestagini, Paolo and Tubaro, Stefano},
title = {Learn from Simulations, Adapt to Observations: Super-Resolution of Isoprene Emissions via Unpaired Domain Adaptation},
journal = {Remote Sensing},
volume = {16},
year = {2024},
number = {21},
doi = {10.3390/rs16213963}
}
```
##License

This project is licensed under the MIT License.

## Acknowledgement
- CycleGAN network is built on [PyTorch-CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN) and [pytorch-CycleGAN-and-pix2pix
](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
- SAN network is built on [SAN](https://github.com/daitao/SAN).
- The entire work was supported by the Italian Ministry of University and
Research [MUR](https://www.mur.gov.it/it) and the European Union (EU) under the [PON/REACT](https://www.ponic.gov.it/sites/PON/ReactEU) project.

<img src="./logos/ispl_logo.png" width="110px" alt="ISPL logo"></img>
<img src="./logos/polimi_logo.png" width="230px" alt="Polimi logo"></img>
<img src="./logos/pon_logo.png" width="160px" alt="PON logo"></img>
<img src="./logos/mur_logo.png" width="80px" alt="MUR logo"></img>
<img src="./logos/ue_logo.png" width="110px" alt="UE logo"></img>

##Misc
### :busts_in_silhouette: About Us
- Image and Sound Processing Lab ([ISPL](http://ispl.deib.polimi.it/))
- Department of Electronics, Information and Bioengineering ([DEIB](https://www.deib.polimi.it/eng/home-page))
- [Politecnico di Milano](https://www.polimi.it/en)
- Antonio Giganti, [ResearchGate](https://www.researchgate.net/profile/Antonio-Giganti), [LinkedIn](https://www.linkedin.com/in/antoniogiganti/)
- Sara Mandelli, [ResearchGate](https://www.researchgate.net/profile/Sara-Mandelli), [LinkedIn](https://www.linkedin.com/in/saramandelli/)
- Paolo Bestagini, [LinkedIn](https://www.linkedin.com/in/paolo-bestagini-390b461b4/)
- Stefano Tubaro, [ResearchGate](https://www.researchgate.net/profile/Stefano-Tubaro), [LinkedIn](https://www.linkedin.com/in/stefano-tubaro-73aa9916/)

### :microphone: Podcast Version
A podcast version of the article is available at [https://studio.youtube.com/video/lNCfx9XUou0/edit](https://studio.youtube.com/video/lNCfx9XUou0/edit).
### :bookmark_tabs: Our BVOC Related Works
A collection of the works from our team focused on BVOC emissions are available at [https://github.com/polimi-ispl/sr-bvoc](https://github.com/polimi-ispl/sr-bvoc).
