# Counting penguins with density map estimation 

## Environment 
- python >=3.6
- pytorch >=1.0
- opencv-python >=4.0
- scipy >=1.4.0
- tqdm
- h5py >=2.10
- pillow >=7.0.0
- pandas

## Construct dataset from annotation
With all the original data (Four folders: Thomas, Jack, Maisie, Luke), and annotations(JSON folder) in the same root, simply run the following code. This code file includes the way to generate density maps.

```python generate_dataset.py ```<br />

Then split the dataset into the train, Val and test sets.

```python train_val_test.py ```<br />

If everything works well, you will get a new folder called 'processed_data' under the current root.


## Train the model
```python train.py ```<br />

## Test the model
```python test.py ```<br />

We provide a pre-trained model and it can be downlowded here: https://drive.google.com/file/d/1wzlCivbuNlJ8BICXXlrb6mxajBgAzq7z/view?usp=sharing
