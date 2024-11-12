# Story Point Estimator

This repository contains code for training and evaluating and exporting models for story point estimation.

The dataset used for training is the [IEEE TSE2018 dataset](https://github.com/jai2shukla/JIRA-Estimation-Prediction/tree/master/storypoint/IEEE%20TSE2018/dataset) which contains user stories from 16 open-source projects from 9 different organizations.


Currently, the following models are supported:
- [x] [DistilBERT](https://arxiv.org/abs/1910.01108)


## Setup

#### Create and activate conda environment

```bash
conda create --name spestimator python=3.8
conda activate spestimator
```

#### Install dependencies

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

#### Setup dotenv

Create a `.env` file in the root directory and add the following environment variables:

```bash
DATA_PATH=<path to data directory>
```


## Usage

#### Train models for story point estimation

See training options

```bash
python train.py --help
```

Train a model with your desired options while also seing validation results

```bash
python train.py <options>
```

Users can see in the experiments folder the results of their experiments in a folder with the experiment's name. The folder contains:
        - a checkpoints folder with the model's saved checkpoints
        - a params.json file with the experiment's parameters
        - a log file with the experiment's logs regarding the training and validation results


#### Evaluate existing models

See evaluation options

```bash
python eval.py --help
```

Evaluate a model with your desired options

```bash
python eval.py <options>
```


#### Export models

See export options

```bash
python export.py --help
```

Export a model with your desired options

```bash
python export.py <options>
```

Exports the model and tokenizer to the specified directory. The model is stored in a .safetensors file and the tokenizer in a vocab.txt file and 4 json files: config.json, special_tokens_map.json, tokenizer_config.json, and tokenizer.json.