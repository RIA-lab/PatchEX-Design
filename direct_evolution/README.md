### Code for HADES

This repository contains code to reproduce the core results in paper.

1. prepare for conda environment:
```
conda env create -f environment.yml
```

2. run.sh provide scripts for model training
```
bash run.sh
```

3. check log files in ```log``` folder for training states

4. after training finished, check ```results``` folder

#### Note: to accelerate model training and evaluation, over 300K predicted structures for all GB1 and PhoQ variants will be public available upon acceptance. Predicted structures should be placed in ```pdb_cache``` folder. For fast verification in review process, the structure-related module is disabled by default.
