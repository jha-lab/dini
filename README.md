# DINI

Data Imputation using Neural Inversion for Edge Applications

## Result reproduction

To generate corrupt data:
```python
python3 corrupt.py --dataset <dataset> --strategy <strategy>
```
where `<dataset>` can either be `breast`, `diabetes`, `diamonds`, `energy`, `flights`, or `yacht`. The flag `<strategy>` can be any one of `MCAR`, `MAR`, `MNAR`, `MSAR`, or `MPAR`.

To run DINI model:
```python
python3 dini.py --model <model> --dataset <dataset> --retrain
```
where `<model>` can either be `FCN`, `FCN2`, `LSTM2`, or `TXF2`. To model uncertainties using an MC dropout layer, use the flag `--model_unc`. You can also define the fraction of to start imputing on using `--impute_fraction <fracion>`, where `<fraction>` is a number between `0` and `1`.

To run imputation using all baselines, including DINI:
```python
python3 impute.py --dataset <dataset> --strategy <strategy>
```

To run surrogate modeling on imputed data, for three case studies:
```python
python3 model.py --dataset <case_dataset> --strategy <strategy>
```
where `<case_dataset>` can either be `gas`, `swat`, or `covid_cxr`. Note that `swat` dataset is not public and will have to be downloaded into `data/swat` directory.