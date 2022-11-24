# DINI: Data Imputation using Neural Inversion for Edge Applications

![Python Version](https://img.shields.io/badge/python-v3.9-blue)
![Conda](https://img.shields.io/badge/conda%7Cconda--forge-v4.12.0-blue)
![PyTorch](https://img.shields.io/badge/pytorch-v1.11.0-e74a2b)
![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FJHA-Lab%2Fdini&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)

DINI is a tool to impute tabular **multi-input/multi-output data** that can have features that are **continuous, categorical, or a combination thereof**. DINI takes in data with missing values, and iteratively imputes it while training a surrogate model that could be leveraged for downstream tasks. This facilitates **machine learning with corrupted/missing data** by state-of-the-art imputation. It works with **any dataset** and **any PyTorch model**.

## Table of Contents
- [Environment Setup](#environment-setup)
  - [Clone this repository](#clone-this-repository)
  - [Setup python environment](#setup-python-environment)
- [Replicating results](#replicating-results)
- [Hacking DINI](#hacking-dini)
- [License](#license)

## Environment setup

### Clone this repository

```shell
git clone --recurse-submodules https://github.com/jha-lab/dini.git
cd dini
```

### Setup python environment  

The python environment setup is based on conda. The script below creates a new environment named `dini` or updates an existing environment on the macOS-arm64 platform:
```shell
source setup/env_step.sh
```

For any other platform, you can use the environment files. For pip installation:
```shell
pip install --requirement setup/requirements.txt
```
For conda installation:
```shell
conda env create --file setup/environment.yaml
conda activate dini
```

## Replicating results

To generate corrupt data:
```python
python3 corrupt.py --dataset <dataset> --strategy <strategy>
```
where `<dataset>` can either be `breast`, `diabetes`, `diamonds`, `energy`, `flights`, or `yacht`. The flag `<strategy>` can be any one of `MCAR`, `MAR`, `MNAR`, `MSAR`, or `MPAR`.

To run DINI model:
```python
python3 dini.py --model <model> --dataset <dataset> --retrain
```
where `<model>` can either be `FCN`, `FCN2`, `LSTM2`, or `TXF2`. The one used in the paper is `FCN2`. To model uncertainties using an MC dropout layer, use the flag `--model_unc`. You can also define the fraction to start imputing on using `--impute_fraction <fracion>`, where `<fraction>` is a number between `0` and `1` (see Table 3 in the paper).

To run imputation using all baselines, including DINI:
```python
python3 impute.py --dataset <dataset> --strategy <strategy>
```

To run surrogate modeling on imputed data, for three case studies:
```python
python3 model.py --dataset <case_dataset> --strategy <strategy>
```
where `<case_dataset>` can either be `gas`, `swat`, or `covid_cxr`. Note that `swat` dataset is not public and will have to be downloaded into `data/swat/` directory. To do this, get access to the dataset using this [link](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/). Then, save `SWaT_Dataset_Attack_v0.csv` to `data/swat/` directory.

## Hacking DINI

To run any PyTorch model, you can modify the `src/models.py` file. See examples (namely models `FCN`, `FCN2`, `LSTM2`, or `TXF2`) in that file. To use any dataset, convert it to a `data.csv` file, placed in `data/<dataset>` directory. Then, the following lines can be added to the `process` function in `corrupt.py`:
```python
elif dataset == <dataset>:
	def split(df):
		return df.iloc[:, :-<out_col>].values, df.iloc[:, -<out_col>:].values
```
where `<dataset>` is the name of the dataset, and `<out_col>` is the number of output columns in the chosen dataset. 

## Developer

[Shikhar Tuli](https://github.com/shikhartuli). For any questions, comments or suggestions, please reach me at [stuli@princeton.edu](mailto:stuli@princeton.edu).

## Cite this work

Cite our work using the following bitex entry:
```
@article{tuli2022sr,
      title={{DINI}: Data Imputation using Neural Inversion for Edge Applications}, 
      author={Tuli, Shikhar and Jha, Niraj K.},
      journal={Scientific Reports},
      volume={12},
      pages={20210},
      year={2022},
      publisher={Nature Publishing Group}
}
```

## License

BSD-3-Clause. 
Copyright (c) 2021, Shikhar Tuli and JHA-Lab.
All rights reserved.

See License file for more details.
