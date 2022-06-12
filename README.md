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

## Folder organisation

```
root
 ├─ project_example.ipynb (main script)        
 ├─ environment.yaml    
 ├─ README.md    
 ├─ scripts
 │   ├─ data_preparation_support.py
 │   ├─ plotting_support.py
 │   ├─ prepare_input_stats.py
 │   ├─ prepare_vectors.py
 │   ├─ shorten_encoding.py
 │   ├─ test_learning.py     
 │   └─ model_scripts
 │       ├─ decision_trees.py
 │       ├─ deep_learning.py
 │       ├─ mlp.py
 │       ├─ nearest_neighbours.py
 │       └─ random_tree.py
 │
 ├─ data
 │   ├─ clean_dataset.pkl
 │   ├─ clean_dataset_biovec.pkl
 │   ├─ clean_dataset_original.pkl
 │   ├─ clean_dataset_singletons.pkl
 │   ├─ clean_dataset_triplets.pkl
 │   ├─ data_file.fasta
 │   ├─ clustering 
 │   │   └─ [Clustering temp files]
 │   │
 │   ├─ full
 │   │   └─ [Full Uniprot file -- to download]
 │   │
 │   └─ vectors
 │       ├─ [Temp biovec vector files]
 │       └─ class_folder
 │           └─ [Temp class files]
 │
 ├─ report
 ├─ presentation
 ├─ CD-HIT
 └─ BIOVEC
 
```

## References
- Data source - Uniprot
  - The UniProt Consortium, UniProt: the universal protein knowledgebase in 2021, Nucleic Acids Research, Volume 49, Issue D1, 8 January 2021, Pages D480–D489, https://doi.org/10.1093/nar/gkaa1100
- Biovec
  - Article Source: Continuous Distributed Representation of Biological Sequences for Deep Proteomics and Genomics
  - Original by: https://github.com/kyu999/biovec
  - Because of installation problems with raw package, I used repository implementing protvec under open licence: https://gitlab.com/victiln/protvec/-/tree/master/source
- CD-HIT
  - Li W, Godzik A. Cd-hit: a fast program for clustering and comparing large sets of protein or nucleotide sequences. Bioinformatics. 2006 Jul 1;22(13):1658-9. doi: 10.1093/bioinformatics/btl158. Epub 2006 May 26. PMID: 16731699.