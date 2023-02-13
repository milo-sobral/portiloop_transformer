import pathlib
import random
from torch.utils.data import DataLoader
from transformiloop.src.data.pretraining import read_pretraining_dataset
from transformiloop.src.data.spindle_trains import EquiRandomSampler, SpindleTrainDataset, read_spindle_trains_labels
from transformiloop.src.utils.configs import initialize_config


ds_dir = pathlib.Path(__file__).parents[2].resolve() / 'dataset'
MASS_dir = ds_dir / 'MASS_preds'
# Read all the subjects available in the dataset
labels = read_spindle_trains_labels(ds_dir) 

config = initialize_config("test")
config['model_type'] = "transformer"
config['classes'] = 2

# Divide the subjects into train and test sets
subjects = list(labels.keys())
# random.shuffle(subjects)
train_subjects = subjects[:2]

# Read the pretraining dataset
data = read_pretraining_dataset(MASS_dir, patients_to_keep=train_subjects)

# Create the train and test datasets
train_dataset = SpindleTrainDataset(train_subjects, data, labels, config)

test_dataloader = DataLoader(
    train_dataset,
    batch_size=config['batch_size_validation'],
    sampler=EquiRandomSampler(train_dataset, sample_list=[1, 2]),
    pin_memory=True,
    drop_last=True
)

for i in enumerate(test_dataloader):
    pass

print("Done")
