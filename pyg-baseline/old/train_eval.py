import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from tqdm import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cross_validation_with_val_set(dataset, model, epochs, batch_size, lr):

    # batch_size = len(dataset.data.x)
    # print("batch_size: ", batch_size)
    labels = torch.ones(batch_size).long().to(device)

    test_dataset = dataset[:]
    train_dataset = dataset[:]

    if 'adj' in test_dataset[0]:
        train_loader = DenseLoader(train_dataset, batch_size, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size, shuffle=False)

    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=lr)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    for _ in tqdm(range(1, epochs + 1)):
        train(model, optimizer, train_loader, labels)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    print("PyG Avg Time (ms): {:.3f}".format((t_end - t_start)/epochs*1e3))

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def train(model, optimizer, loader, labels):
    model.train()
    
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        # sys.exit(0)
        labels = torch.ones(out.size(0)).long().to(device)   
        # print(out.size())
        # print(labels.size())
        loss = F.nll_loss(out, labels.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)