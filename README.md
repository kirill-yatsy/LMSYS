[![Code-Generator](https://badgen.net/badge/Template%20by/Code-Generator/ee4c2c?labelColor=eaa700)](https://github.com/pytorch-ignite/code-generator)

# Text Classification Template

This is the text classification template by Code-Generator using `bert-base-uncased` model from HuggingFace Transformers and `imdb` dataset from HuggingFace datasets and training is powered by PyTorch and PyTorch-Ignite.

## Getting Started

Install the dependencies with `pip`:

```sh
pip install -r requirements.txt --progress-bar off -U
```

### Code structure

```
|
|- README.md
|
|- main.py : main script to run
|- data.py : helper module with functions to setup input datasets and create dataloaders
|- models.py : helper module with functions to create a model or multiple models
|- trainers.py : helper module with functions to create trainer and evaluator
|- utils.py : module with various helper functions
|- requirements.txt : dependencies to install with pip
|
|- config.yaml : global configuration YAML file
```

## Training

### 1 GPU Training

```sh
python main.py config.yaml
```
