# DINI
Data Imputation using Neural Inversion

# To Run

To generate corrupt data
```python
python3 corrupt.py --dataset MSDS
```

To run model
```python
python3 main.py --model FCN2 --dataset MSDS --retrain
```