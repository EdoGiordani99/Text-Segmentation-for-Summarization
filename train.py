from params import params
from utils import plot_train_eval
from model import SegmentationModel
from trainer import SegmentationTrainer
from torch.utils.data import DataLoader
from dataset import SegmentationDataset, collate_fn


if __name__ == '__main__':
    """
    Train the model by running in terminal: python train.py
    """
    train_dataset = SegmentationDataset('data/texts', 'data/train_annotations.txt')
    valid_dataset = SegmentationDataset('data/texts', 'data/valid_annotations.txt')

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=params['data']['batch_size'],
                                  shuffle=params['data']['shuffle'],
                                  collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=params['data']['batch_size'],
                                  shuffle=params['data']['shuffle'],
                                  collate_fn=collate_fn)

    model = SegmentationModel(params['model'])

    trainer = SegmentationTrainer(model=model,
                                  train_dataloader=train_dataloader,
                                  valid_dataloader=valid_dataloader,
                                  params=params,
                                  compute_f1=True)

    history = trainer.train()

    plot_train_eval(history)
