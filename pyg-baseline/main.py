from itertools import product
import argparse
from datasets import get_dataset
from train_eval import cross_validation_with_val_set

from gcn import GCN
# from graph_sage import GraphSAGE
# from gin import GIN0

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--dataset', type=str, default='COLLAB')
parser.add_argument('--net', type=str, default='gcn')
args = parser.parse_args()

nets = None
if args.net == 'gcn':
    layers = [2]
    hiddens = [16]
    nets = [GCN]

# if args.net == "graphsage":
#     layers = [2]
#     hiddens = [16]
#     nets = [GraphSAGE]

# if args.net == 'gin':
#     layers = [5]
#     hiddens = [64]
#     nets = [GIN0]

datasets = [args.dataset]

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print('{:02d}/{:03d}: Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(
        fold, epoch, val_loss, test_acc))

results = []
for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    print('{} - {}\n-----'.format(dataset_name, Net.__name__))

    for num_layers, hidden in product(layers, hiddens):
        dataset = get_dataset(dataset_name, sparse=True, cleaned=True)
        model = Net(dataset, num_layers, hidden)
        # loss, acc, std = cross_validation_with_val_set(
        #     dataset,
        #     model,
        #     folds=10,
        #     epochs=args.epochs,
        #     batch_size=args.batch_size,
        #     lr=args.lr,
        #     lr_decay_factor=args.lr_decay_factor,
        #     lr_decay_step_size=args.lr_decay_step_size,
        #     weight_decay=0,
        #     logger=None,
        # )
        # if loss < best_result[0]:
        #     best_result = (loss, acc, std)

        cross_validation_with_val_set(
                dataset,
                model,
                folds=10,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                lr_decay_factor=args.lr_decay_factor,
                lr_decay_step_size=args.lr_decay_step_size,
                weight_decay=0,
                logger=None,
            )

    # desc = '{:.3f} ± {:.3f}'.format(best_result[1], best_result[2])
    # print('Best result - {}'.format(desc))
    # results += ['{} - {}: {}'.format(dataset_name, model, desc)]
# print('-----\n{}'.format('\n'.join(results)))
print()
# print()