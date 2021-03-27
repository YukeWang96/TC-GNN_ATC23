import argparse
from datasets import get_dataset
from train_eval import cross_validation_with_val_set

from gnn import GCN, AGNN

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--dataset', type=str, default='PROTEINS_full')
parser.add_argument('--net', type=str, default='agnn', choices=['gcn', 'agnn'])
parser.add_argument('--num_layer', type=int, default=4, help="number of Graph Conv Layers")
parser.add_argument('--hidden', type=int, default=16, help="hidden dimension size of Conv Layers")
args = parser.parse_args()
print(args)

dataset = get_dataset(args.dataset, sparse=True, cleaned=True)
if args.net == 'gcn':
    model = GCN(dataset, args.num_layer, args.hidden)
if args.net == "agnn":
    model = AGNN(dataset, args.num_layer, args.hidden)
print(model)
cross_validation_with_val_set(dataset, model, epochs=args.epochs, \
                                batch_size=args.batch_size, lr=args.lr)
print()