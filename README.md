# DINI
Data Imputation using Neural Inversion

# To Run

To generate corrupt data
```python
python3 corrupt.py --dataset MSDS --strategy MNAR
```

To run DINI model
```python
python3 dini.py --model FCN2 --dataset MSDS --retrain
```
