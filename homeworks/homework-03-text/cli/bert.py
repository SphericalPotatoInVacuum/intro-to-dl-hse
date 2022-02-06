from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from cli.regressors import regressors


class MyDataSet(Dataset):
    def __init__(self, X, y=None):
        self.positive = X['positive'].tolist()
        self.negative = X['negative'].tolist()
        self.targets = y.tolist() if y is not None else [0] * len(self.positive)

    def __getitem__(self, i):
        return self.positive[i], self.negative[i], self.targets[i]

    def __len__(self):
        return len(self.positive)


class CustomBERTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=1
        )
        # self.bert.classifier = nn.Linear(768, 1)
        # for param in self.bert.distilbert.parameters():
        #     param.requires_grad = False

    def forward(self, ids, attention_mask):
        output = self.bert(
            ids,
            attention_mask=attention_mask
        )
        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        return output.logits


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        positive = [item[0] for item in batch]
        negative = [item[1] for item in batch]
        targets = [item[2] for item in batch]
        outputs = tokenizer.batch_encode_plus(
            list(zip(positive, negative)),
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=100,
            return_tensors='pt'
        )
        outputs['targets'] = torch.FloatTensor(targets).reshape((-1, 1))
        return outputs
    return collate_fn


class BertRegressor(BaseEstimator, RegressorMixin):
    need_preprocessing = False

    def __init__(self, epochs=1):
        self.epochs = epochs
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def fit(self, X, y=None):
        self.model = CustomBERTModel().cuda()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-4,
        )
        scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
        criterion = nn.L1Loss()
        data_loader = DataLoader(
            MyDataSet(X, y),
            batch_size=32,
            num_workers=4,
            shuffle=True,
            collate_fn=make_collate_fn(self.tokenizer)
        )
        self.model.train()
        losses = []
        point_batches = 1000 // data_loader.batch_size
        for epoch in range(self.epochs):
            pb = tqdm(data_loader)
            avg_mae = 0
            batches = 0
            for batch in pb:
                input_ids = batch['input_ids'].cuda()
                targets = batch['targets'].cuda()
                attention_mask = batch['attention_mask'].cuda()

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                avg_mae += loss.item()
                batches += 1
                if batches % point_batches == 0:
                    avg_mae /= point_batches
                    scheduler.step(avg_mae)
                    pb.set_description(f'MAE: {avg_mae:.5f}')
                    avg_mae = 0
                    batches = 0
        plt.figure()
        plt.plot(np.arange(len(losses)), losses)
        plt.savefig('loss.png')
        return self

    def predict(self, X, y=None):
        data_loader = DataLoader(
            MyDataSet(X),
            batch_size=100,
            collate_fn=make_collate_fn(self.tokenizer)
        )
        self.model.eval()
        outputs = torch.Tensor()
        with torch.no_grad():
            for batch in tqdm(data_loader):
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                outputs = torch.cat((outputs, self.model(input_ids, attention_mask=attention_mask).cpu()), 0)
        return outputs


regressors['BERT'] = BertRegressor
