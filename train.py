import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import trange
import argparse

from custom_dataset import CustomDataset
from custom_evaluate import CustomEvaluator
from fddbenchmark import FDDDataset, FDDDataloader, FDDEvaluator
from gnn import GNN_TAM


def parse_args():
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--dataset', type=str, default='reinartz_tep')
    parser.add_argument('--n_epochs', type=int, default=40)
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_gnn', type=int, default=1)
    parser.add_argument('--gsl_type', type=str, default='tanh')
    parser.add_argument('--n_hidden', type=int, default=1024)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--name', type=str, default='gnn')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def train():
    args = parse_args()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = args.device
    print('Using device:', device)
    print('Dataset:', args.dataset)
    # Data preparation:
    if args.dataset in ['reinartz_tep', 'rieth_tep', 'small_tep']:
        dataset = FDDDataset(name=args.dataset)
        scaler = StandardScaler()
        scaler.fit(dataset.df.loc[dataset.train_mask])
        dataset.df[:] = scaler.transform(dataset.df)

        train_dl = FDDDataloader(
            dataframe=dataset.df,
            label=dataset.label,
            mask=dataset.train_mask,
            window_size=args.window_size,
            step_size=args.step_size,
            use_minibatches=True,
            batch_size=args.batch_size,
            shuffle=True
        )

        test_dl = FDDDataloader(
            dataframe=dataset.df,
            label=dataset.label,
            mask=dataset.test_mask,
            window_size=args.window_size,
            step_size=args.step_size,
            use_minibatches=True,
            batch_size=args.batch_size,
            shuffle=True
        )
    elif args.dataset in ['IoT_Modbus', 'IoT_Weather', 'mhealth', 'pamap2']:
        dataset = CustomDataset(dataset=args.dataset, window_size=args.window_size, step_size=args.step_size)
        train_dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)

        test_dataset = CustomDataset(dataset=args.dataset, window_size=args.window_size, step_size=args.step_size, mode='test')
        test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    else:
        raise NotImplemented(f'{args.dataset} is not supported')
    n_nodes = dataset.df.shape[1]
    n_classes = len(set(dataset.label))

    # Model creation:
    model = GNN_TAM(n_nodes=n_nodes,
                    window_size=args.window_size,
                    n_classes=n_classes,
                    n_gnn=args.n_gnn,
                    gsl_type=args.gsl_type,
                    n_hidden=args.n_hidden,
                    alpha=args.alpha,
                    k=args.k,
                    device=device)
    model.to(device)
    # Training:
    optimizer = Adam(model.parameters(), lr=0.001)
    weight = torch.ones(n_classes) * 0.5
    weight[1:] /= (n_classes - 1)
    outer_bar = trange(args.n_epochs, desc="Epoch: 0, ...")
    for e in outer_bar:
        av_loss = []
        model.train()
        for train_ts, train_index, train_label in train_dl:
            ts = torch.FloatTensor(train_ts).to(device, non_blocking=True)
            ts = torch.transpose(ts, 1, 2)
            train_label = torch.LongTensor(train_label).to(device, non_blocking=True)
            logits = model(ts)
            loss = F.cross_entropy(logits, train_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            av_loss.append(loss.item())

        # model.eval()
        # preds = []
        # test_labels = []
        # for test_ts, test_index, test_label in test_dl:
        #     ts = torch.FloatTensor(test_ts).to(device)
        #     ts = torch.transpose(ts, 1, 2)
        #     with torch.no_grad():
        #         logits = model(ts)
        #     pred = logits.argmax(axis=1).cpu().numpy()
        #     if isinstance(test_index, torch.Tensor):
        #         test_index = test_index.numpy()
        #     preds.append(pd.Series(pred, index=test_index))
        #     test_labels.append(pd.Series(test_label, index=test_index))
        # pred = pd.concat(preds)
        # test_label = pd.concat(test_labels)
        #
        # evaluator = CustomEvaluator(step_size=args.step_size)
        # # evaluator = FDDEvaluator(step_size=1)
        # # evaluator.print_metrics(test_label, pred)
        # metrics = evaluator.evaluate_classification(test_label, pred)
        #
        # print(f'Epoch: {e + 1:2d}/{args.n_epochs}, average CE loss: {sum(av_loss) / len(av_loss):.4f}, {metrics}')

        # outer_bar.update(1)
        outer_bar.set_description(f'Epoch: {e + 1:2d}/{args.n_epochs}, average CE loss: {sum(av_loss) / len(av_loss):.4f}')

    model.eval()
    preds = []
    test_labels = []
    for test_ts, test_index, test_label in test_dl:
        ts = torch.FloatTensor(test_ts).to(device)
        ts = torch.transpose(ts, 1, 2)
        with torch.no_grad():
            logits = model(ts)
        pred = logits.argmax(axis=1).cpu().numpy()
        if isinstance(test_index, torch.Tensor):
            test_index = test_index.numpy()
        preds.append(pd.Series(pred, index=test_index))
        test_labels.append(pd.Series(test_label, index=test_index))
    pred = pd.concat(preds)
    test_label = pd.concat(test_labels)

    # evaluator = CustomEvaluator(step_size=args.step_size)
    evaluator = FDDEvaluator(step_size=1)
    evaluator.print_metrics(test_label, pred)
    # metrics = evaluator.evaluate_classification(test_label, pred)

    print(f'Epoch: {e + 1:2d}/{args.n_epochs}, average CE loss: {sum(av_loss) / len(av_loss):.4f}')

    torch.save(model.state_dict(), 'saved_models/' + args.name + str(args.n_gnn) + 'x' + str(args.n_hidden) + '_' + args.gsl_type + '_' + args.dataset + '.pt')


if __name__ == '__main__':
    train()
