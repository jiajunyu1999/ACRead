import random
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

import torch
import torch.optim as optim
import torch.nn.functional as F
import pickle
from gnn import GNN
from utils import get_kfold_idx_split, str2bool

import networkx as nx
import argparse
import time
import numpy as np

from torch_geometric.utils import to_networkx
import os
from tqdm import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from torch_geometric.utils import to_networkx
import os
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from sklearn.metrics import roc_auc_score, average_precision_score

mcls_criterion = torch.nn.CrossEntropyLoss()

def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    

def save_graph(batch, step):  
    # Convert the graph to a NetworkX graph
    G = to_networkx(batch, node_attrs=['x'], edge_attrs=['edge_attr'], to_undirected=True)
    
    # Create the plot
    plt.figure(figsize=(8, 8))
    
    # Draw the graph with node labels as their IDs
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_weight='bold', node_size=500, font_size=10)
    
    # Get initial x values for each node
    initial_x = {n: data['x'] for n, data in G.nodes(data=True)}
    
    # Save initial x values and edge weights in a text file
    with open(f'./figure/graph_data_step_{step}.txt', 'w') as f:
        f.write(f'label: {batch.y}\n')
        for node, x_value in initial_x.items():
            f.write(f'Node {node}: {x_value}\n')
        f.write("\nEdge weights:\n")
        print(G)

    # Save the figure
    plt.savefig(f'./figure/graph_step_{step}.png')
    plt.close()

def train(model, device, loader, optimizer_gnn, optimizer_seg, args): 
    model.train()
    all_loss = 0
    if not os.path.exists('./figure'):
        os.makedirs('./figure')
        
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch)
        if args.dataset.count('ogbg') != 0: y = batch.y.squeeze(1)
        else: y = batch.y
        is_labeled = y == y
        
        optimizer_gnn.zero_grad()
        
        pred_loss = mcls_criterion(pred.to(torch.float32)[is_labeled], y[is_labeled])
        pred_loss.backward(retain_graph=True)
        
        optimizer_gnn.step()

        if args.read_op == 'sread':
            optimizer_seg.zero_grad()
            align_loss = model.get_aligncost(batch)
            align_loss.backward(retain_graph=True)
            optimizer_seg.step()

        all_loss += pred_loss.item()

        # Visualize and save the graph at each step
        # save_graph(batch, step)
        
    return all_loss


def eval(model, device, loader):
    model.eval()
    
    y_true, y_pred, graph_sizes = [], [], []
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)
            pred = torch.max(pred, dim=1)[1]
                
        y_true.append(batch.y.view(pred.shape))
        y_pred.append(pred)
    
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    correct = y_true == y_pred
    try:
        auc = roc_auc_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        ap = average_precision_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())    
    except:
        auc = 0
        ap = 0
    return {'auc': correct.sum().item()/correct.shape[0], 'auc':auc, 'ap':ap, 'acc': correct.sum().item()/correct.shape[0], 'sizes': graph_sizes, 'y_true': y_true.cpu(), 'y_pred': y_pred.cpu()}
    

def get_centrality(graph):
    G = to_networkx(graph)
    nodes = sorted(G.nodes())

    degrees = torch.Tensor(list(dict(G.degree).values())) / 2
    degree_centrality = nx.degree_centrality(G)
    degree_vector = torch.FloatTensor([degree_centrality[node] for node in nodes])

    betweenness_centrality = nx.betweenness_centrality(G)
    betweenness_vector = torch.FloatTensor([betweenness_centrality[node] for node in nodes])

    closeness_centrality = nx.closeness_centrality(G)
    closeness_vector = torch.FloatTensor([closeness_centrality[node] for node in nodes])

    # eigenvector_centrality = nx.eigenvector_centrality(G)
    # eigenvector_vector = torch.FloatTensor([eigenvector_centrality[node] for node in nodes])

    # katz_centrality = nx.katz_centrality(G, alpha=0.1)
    # katz_vector = torch.FloatTensor([katz_centrality[node] for node in nodes])

    pagerank_centrality = nx.pagerank(G)
    pagerank_vector = torch.FloatTensor([pagerank_centrality[node] for node in nodes])

    hubs, authorities = nx.hits(G)
    hubs_vector = torch.FloatTensor([hubs[node] for node in nodes])
    authorities_vector = torch.FloatTensor([authorities[node] for node in nodes])

    load_centrality = nx.load_centrality(G)
    load_vector = torch.FloatTensor([load_centrality[node] for node in nodes])

    harmonic_centrality = nx.harmonic_centrality(G)
    harmonic_vector = torch.FloatTensor([harmonic_centrality[node] for node in nodes])

    bridges = list(nx.bridges(G.to_undirected()))
    bridge_set = set()
    bridge_feat = torch.zeros((degrees.shape[0],1)).reshape(-1)
    for u, v in bridges:
        bridge_set.add(u)
        bridge_set.add(v)
    bridge_feat[np.array(list(bridge_set))] = 1
    bridge_feat = torch.FloatTensor(bridge_feat)
    features = torch.stack([degree_vector, betweenness_vector, closeness_vector,
                            pagerank_vector, hubs_vector, authorities_vector, load_vector,
                            harmonic_vector, bridge_feat
                            ], dim=1)
    return features

class AddGraphIdTransform:  
    def __init__(self):
        self.graph_id = 0
        
    def __call__(self, data):
        nodes = data.num_nodes
        centrality = get_centrality(data)
        data.nodes = nodes
        data.graph_id = self.graph_id
        self.graph_id += 1
        data.centrality = centrality
        return data

def label_distribution(args):
    from collections import Counter
    print(args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    transform = AddGraphIdTransform()

    if args.dataset.count('ogbg') != 0:
        from ogb.graphproppred import PygGraphPropPredDataset
        dataset = PygGraphPropPredDataset(name='ogbg-molhiv')
        graph_labels = dataset.data.y.view(-1).tolist()
    else:
        dataset = TUDataset(root = './new_data', name = args.dataset, pre_transform=transform)
        num_nodefeats = max(1, dataset.num_node_labels)
        num_edgefeats = max(1, dataset.num_edge_labels)
        print('end')
        graph_labels = [data.y.item() for data in dataset]

    label_counts = Counter(graph_labels)
    total_labels = sum(label_counts.values())
    label_ratios = {label: count / total_labels for label, count in label_counts.items()}
    print("Label Ratios:")
    for label, ratio in label_ratios.items():
        print(f"Dataset {args.dataset} Label {label}: {ratio:.2%}")


def main(args):
    
    print(args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    transform = AddGraphIdTransform()
    
    if args.dataset.count('ogbg') != 0:
        from ogb.graphproppred import PygGraphPropPredDataset
        dataset = PygGraphPropPredDataset(name = 'ogbg-molhiv') 
        num_nodefeats = max(1,dataset.x.shape[1])
        num_edgefeats = max(1, dataset.num_edge_features)
    
    else:
        dataset = TUDataset(root = './new_data', name = args.dataset, pre_transform=transform)
        num_nodefeats = max(1, dataset.num_node_labels)
        num_edgefeats = max(1, dataset.num_edge_labels)
    
    
    num_classes = int(dataset.data.y.max()) + 1
    split_idx = get_kfold_idx_split(dataset, num_fold=args.num_fold, random_state=args.seed)
    valid_list, test_list = [], []

    # if args.dataset == 'PROTEINS':
    #     num_nodefeats = dataset.filter_x.shape[1] - 1
    if args.dataset in ['IMDB-BINARY','IMDB-MULTI']:
        num_nodefeats = 10
        
    all_test_sizes, all_test_accuracies = [], []
    for fold_idx in range(10): 
        train_loader = DataLoader(dataset[split_idx["train"][fold_idx]], batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
        valid_loader = DataLoader(dataset[split_idx["augvalid"][fold_idx]], batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
        test_loader = DataLoader(dataset[split_idx["test"][fold_idx]], batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
        model = GNN(gnn_type = args.gnn, num_classes = num_classes, num_nodefeats = num_nodefeats, num_edgefeats = num_edgefeats,
                        num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, 
                        read_op = args.read_op, args = args).to(device)
        
        optimizer_gnn = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        if args.read_op == 'sread':
            optimizer_seg = optim.Adam(model.read_op.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) 
        else:
            optimizer_seg = None
        
        valid_curve, test_curve = [], []
        best_valid_perf, no_improve_cnt = -np.inf, 0
        for epoch in range(1, args.max_epochs + 1):
            loss = train(model, device, train_loader, optimizer_gnn, optimizer_seg, args)
            valid_perf = eval(model, device, valid_loader)
            test_perf = eval(model, device, test_loader)
            print('%3d\t%.6f\t%.6f\t%.6f'%(epoch, valid_perf['acc'], test_perf['acc'], loss))

            valid_curve.append(valid_perf['acc'])
            test_curve.append(test_perf['acc'])

            if no_improve_cnt > args.early_stop:
                break
            elif valid_perf['acc'] > best_valid_perf:
                best_valid_perf = valid_perf['acc']
                no_improve_cnt = 0
            else:
                no_improve_cnt += 1
            
        best_val_epoch = np.argmax(np.array(valid_curve))

        print('%2d-fold Valid\t%.6f'%(fold_idx+1, valid_curve[best_val_epoch]))
        print('%2d-fold Test\t%.6f'%(fold_idx+1, test_curve[best_val_epoch]))

        valid_list.append(valid_curve[best_val_epoch])
        test_list.append(test_curve[best_val_epoch])

        all_test_sizes.extend(test_perf['sizes'])
        all_test_accuracies.extend([1 if y == y_pred else 0 for y, y_pred in zip(test_perf['y_true'], test_perf['y_pred'])])
 
    valid_list = np.array(valid_list)*100
    test_list = np.array(test_list)*100
    print('Valid Acc:{:.4f}, Std:{:.4f}, Test Acc:{:.4f}, Std:{:.4f}'.format(np.mean(valid_list), np.std(valid_list), np.mean(test_list), np.std(test_list)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--gnn', type=str, default='gcn',
                    help='Type of GNN to use: gin, gcn, gat')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='Dropout ratio for GNN layers (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='Number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='Dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--read_op', type=str, default='acread',
                        help='Graph readout operation (e.g., sum, mean) (default: sum)')

    # Optimization parameters
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training (default: 256)')
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='Maximum number of epochs for training (default: 500)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay (L2 regularization) for the optimizer (default: 0.0001)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate for the optimizer (default: 0.001)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of worker threads for data loading (default: 0)')
    parser.add_argument('--early_stop', type=int, default=50,
                        help='Patience for early stopping (number of epochs without improvement) (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID to use for training (default: 0)')

    # Dataset parameters
    parser.add_argument('--datapath', type=str, default="./dataset",
                        help='Path to the directory containing datasets (default: ./dataset)')
    parser.add_argument('--dataset', type=str, default="NCI1",
                        help='Name of the dataset to use (default: NCI1)')
    parser.add_argument('--num_fold', type=int, default=10,
                        help='Number of folds for cross-validation (default: 10)')
    parser.add_argument('--score_nums', type=int, default=9,
                        help='Number of score evaluations (default: 9)')
    parser.add_argument('--head', type=int, default=4,
                        help='Number of attention heads in the model (default: 4)')
    parser.add_argument('--drop', type=float, default=0.5,
                        help='Dropout rate for ACRead attention mechanism (default: 0.5)')
    parser.add_argument('--phi', type=str, default='hadama',
                        help='Function for ACRead g_phi operation (default: hadama)')

    args = parser.parse_args()
    main(args)
    
