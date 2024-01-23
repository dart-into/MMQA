# MMQA
# UIQAVSI
Code for MMQA
![./Net](https://github.com/dart-into/MMQA/blob/main/framework.png)

## Dataset
| Dataset   | Links                                                       |
| --------- | ----------------------------------------------------------- |
| TID2013      | http://www.ponomarenko.info/tid2013.htm     |
| KADID10K      | http://database.mmsp-kn.de/kadid-10k-database.html      |
| LIVE-C      | https://live.ece.utexas.edu/research/ChallengeDB/index.html      |
| LIVE      | https://live.ece.utexas.edu/research/Quality/subjective.htm          |
| CSIQ      | https://s2.smu.edu/~eclarson/csiq.html |
| SPAQ      | https://github.com/h4nwei/SPAQ |
| KonIQ      | http://database.mmsp-kn.de/koniq-10k-database.html |
| AGIQA     |  https://github.com/lcysyzxdxc/AGIQA-3k-Database |
## Requirements
- PyTorch=1.7.0
- Torchvision=0.8.1
- numpy=1.21.6
- scipy=1.7.3
- h5py=3.7.0
- opencv-python =4.7.0.72
## Usages

### Screen out salient regions and non-salient regions of images

If you want to get salient regions and non-salient regions, you need to run screen_salient_data.py.

```
screen_salient_data.py
```

You will get new datasets, and these datasets are inputs of the model.

### Meta training on IQA databases
First you need to modify config paramters to make sure the database path is correct.
Meta training  our model on IQA Dataset.
```
MMQA_newload.py
```
Some available options:
* `--dataset`: Meta training  dataset, support datasets: TID2013 |KADID10K| LIVE | CSIQ | .
* `--lr`: Learning rate.
* `--save_path`: Model and paramter save path.
* `--batch_size`: Batch size.
* `--epochs`:Epochs
* 
```
If you want to repartition the dataset, you'll need to make a new mat file instead.

### Fine-tuning for different datasets
```
FineTune_newload.py
```
Some available options:
* `--dataset_dir`:  Fine-tuning dataset image path.
* `--model_file`: Model and paramter path.
* `--dataset`:  Testing dataset, support datasets:  LIVE-C | SPAQ | KonIQ | CSIQ| AGIQA |.
* `--predict_save_path`: The plcc and srcc are recorded in TID2013_KADID_LIVEC.txt or ew_load_scores.csv.
