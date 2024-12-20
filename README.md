# MMVSR
PyTorch's implementation of "Multi-Memory Streams: A Paradigm for Online Video Super-Resolution in Complex Exposure Scenes".

# Requirements
+ Python 3.8
+ PyTorch 1.8.1 (>1.1.0)
+ cuda 11.3

# Preparing Datasets
Download following datasets (dataset can be download from the official website):
> Training dataset: REDS

> Testing dataset: REDS4, Vid4, Vimeo-90K, UDM10, TOG

Place the datasets in the ./dataset folder, and the detailed path for placing the data can be found in Our_dataloader.py

# Training

```
python Train.py --experiment_indxe '<SAVE_PATH>'
```

# Testing
Pretrain model can be download from the [[url]](https://drive.google.com/file/d/13bKvOJVaZRL9I-wy8Dv2skjsnF3CI9Fn/view?usp=sharing)

Place the Pretrain model in the ./save_model folder
```
python Inference.py --experiment_indxe '<SAVE_PATH>'
```



