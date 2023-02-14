from params import params
from model import SegmentationModel
from torch.utils.data import DataLoader
from dataset import SegmentationDataset, collate_fn

# Dataset
train_dataset = SegmentationDataset('data/texts', 'data/train_annotations.txt')
valid_dataset = SegmentationDataset('data/texts', 'data/valid_annotations.txt')

# Data Loaders
train_dataloader = DataLoader(train_dataset,
                              batch_size=params['data']['batch_size'],
                              shuffle=params['data']['shuffle'],
                              collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_dataset,
                              batch_size=params['data']['batch_size'],
                              shuffle=params['data']['shuffle'],
                              collate_fn=collate_fn)

model = SegmentationModel(params['model'])

# Example
batch = next(iter(train_dataloader))

out = model(batch, compute_preds = True)
print(out)
