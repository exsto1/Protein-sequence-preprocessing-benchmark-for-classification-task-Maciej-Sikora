# Protein sequence preprocessing benchmark for classification task - Maciej Sikora

Package with scripts for the final project for Modeling of Complex Biological Systems subject.

The project focuses on a comparison between original data, naive compression and a more sophisticated biovec model.
To ensure optimal results proposed models are using Grid Search on a selection of parameters.

Because there are multiple models, backed by cross-validation and for multiple datasets, complete runtime can be lengthy.

That's why the proposed sample is not taking the full Swissprot database, but hits from the top biggest families and top most frequent organisms.

However, the provided notebook contains multiple widgets, for easier customization - for example, it is possible to acquire a bigger sample - however please note that the training process is quite resource-heavy.

## Packages

The environment was tested under Ubuntu using Conda with Python 3.8

### Installation process

- Creating a new Conda environment
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

