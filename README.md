## Packages

Environment was tested under Ubuntu using Conda with Python 3.8

### Installation process

- Creating new Conda environment
- Activating env
- Installing Biovec requirements
- Installing package conda requirements
- Upgrading pandas

```
conda create --name benchmark_project_env python=3.8 -y
conda activate benchmark_project_env
pip install -r ./BIOVEC/requirements.txt
conda env update --file environment.yaml
pip install --upgrade numpy scipy pandas
```

