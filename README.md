# Protein sequence preprocessing benchmark for classification task - Maciej Sikora

Package with scripts for final project for Modeling of Complex Biological Systems subject.

Project focuses on comparison between original data, naive compression and more sophisticated biovec model.
To ensure optimal results proposed models are using Grid Search on a selection of parameters.

Because there are multiple models, backed by cross-validation and for multiple datasets, complete runtime can be lengthy.

That's why proposed sample is not taking full Swissprot database, but hits from top biggest families and top most frequent organisms.

However, provided notebook contains multiple widgets, for easier customization - for example it is possible to aquire bigger sample - however please note that training process is quite resource heavy.

## Packages

Environment was tested under Ubuntu using Conda with Python 3.8

### Installation process

- Creating new Conda environment
- Activating env
- Installing Biovec requirements
- Installing package conda requirements - depending on the internet speed, this might take some time (tensorflow)
- Upgrading pandas

```
conda create --name benchmark_project_env python=3.8 -y
conda activate benchmark_project_env
pip install -r ./BIOVEC/requirements.txt
conda env update --file environment.yaml
pip install --upgrade numpy scipy pandas
```

