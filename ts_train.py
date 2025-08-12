import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm.auto import tqdm, trange
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from custom_evaluate import CustomEvaluator
from ts_transformer import TimeSeriesTransformerGSL

def parse_args():
    parser = argparse.ArgumentParser(description='Train transformer+GSL model')
    parser.add_argument('--dataset', type=str, default='mhealth')
    parser.add_argument('--n_epochs', type=int, default=60)
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--name', type=str, default='mhealth_transformer')
    return parser.parse_args()

def train():    
    args = parse_args()
    device = torch.device('cuda:1')
    print('Using device:', device)

    dataset = CustomDataset(dataset=args.dataset, window_size=args.window_size, step_size=args.step_size)
    train_dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)

    test_dataset = CustomDataset(dataset=args.dataset, window_size=args.window_size, step_size=args.step_size, mode='test')
    test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    
    n_nodes = dataset.df.shape[1]
    n_classes = len(set(dataset.label))


    model = TimeSeriesTransformerGSL(
        ts_dim=n_nodes,
        window_size=args.window_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        classes = n_classes,
        device=device
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    model.train()

    for epoch in trange(args.n_epochs, desc="Epochs"):
        epoch_losses = []
        for train_ts, train_index, train_label in train_dl:
            ts = torch.FloatTensor(train_ts).to(device, non_blocking=True)
            train_label = torch.LongTensor(train_label).to(device, non_blocking=True)
            logits = model(ts)
            loss = F.cross_entropy(logits, train_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            model.eval()
            preds = []
            test_labels = []
            for test_ts, test_index, test_label in test_dl:
                ts = torch.FloatTensor(test_ts).to(device)
                with torch.no_grad():
                    logits = model(ts)
                pred = logits.argmax(axis=1).cpu().numpy()
                if isinstance(test_index, torch.Tensor):
                    test_index = test_index.numpy()
                preds.append(pd.Series(pred, index=test_index))
                test_labels.append(pd.Series(test_label, index=test_index))
            pred = pd.concat(preds)
            test_label = pd.concat(test_labels)

            evaluator = CustomEvaluator(step_size=args.step_size)
            metrics = evaluator.evaluate_classification(test_label, pred)

            print(f'Epoch: {epoch + 1:2d}/{args.n_epochs}, average CE loss: {sum(epoch_losses) / len(epoch_losses):.4f}, {metrics}')

    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), f"saved_models/{args.name}.pt")


if __name__ == '__main__':
    train()
