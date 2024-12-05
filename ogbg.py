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
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
mcls_criterion = torch.nn.CrossEntropyLoss()

def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    

def train(model, device, loader, optimizer_gnn, optimizer_seg, args):
    model.train()
    all_loss = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch)
        if args.dataset.count('ogbg') != 0: y = batch.y.squeeze(1)
        else: y = batch.y
        is_labeled = y == y
        optimizer_gnn.zero_grad()
        
        pred_loss = mcls_criterion(pred.to(torch.float32)[is_labeled], y[is_labeled])
        pred_loss.backward(retain_graph = True)
        optimizer_gnn.step()

    

        if args.read_op == 'sread':
            optimizer_seg.zero_grad()
            align_loss = model.get_aligncost(batch)
            align_loss.backward(retain_graph=True)
            optimizer_seg.step()

        all_loss += pred_loss.item()
    return all_loss

from sklearn.metrics import roc_auc_score, average_precision_score
def eval(model, device, loader):
    model.eval()
    
    y_true, y_pred = [], []
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            pred= model(batch)
            pred = torch.max(pred, dim=1)[1]
                
        y_true.append(batch.y.view(pred.shape))
        y_pred.append(pred)
    
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    correct = y_true == y_pred
    auc = roc_auc_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    ap = average_precision_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())    
    return {'auc': correct.sum().item()/correct.shape[0], 'auc':auc, 'ap':ap}

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


def main(args):
    
    print(args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    transform = AddGraphIdTransform()

    if args.dataset.count('ogbg') != 0:


        dataset = PygGraphPropPredDataset(name = args.dataset, pre_transform = transform) 
        # num_nodefeats = 100
        # num_edgefeats = 100
        num_nodefeats = max(1,dataset.x.shape[1])
        num_edgefeats = max(1, dataset.num_edge_features)
    else:
            
        # if args.dataset in ['MUTAG','DD','PROTEINS','NCI1','Mutagencity','IMDB-BINARY','IMDB-MULTI', 'COLLAB']:
        dataset = TUDataset(root = './new_data', name = args.dataset, pre_transform=transform)
        num_nodefeats = max(1, dataset.num_node_labels)
        num_edgefeats = max(1, dataset.num_edge_labels)

    num_classes = int(dataset.data.y.max()) + 1
    split_idx = get_kfold_idx_split(dataset, num_fold=args.num_fold, random_state=args.seed)
    valid_list, test_list = [], []
    times = []

    # if args.dataset == 'PROTEINS':
    #     num_nodefeats = dataset.filter_x.shape[1] - 1
    if args.dataset in ['IMDB-BINARY','IMDB-MULTI']:
        num_nodefeats = 10
        
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
        t = time.time()
        for epoch in range(1, args.max_epochs + 1):
            loss = train(model, device, train_loader, optimizer_gnn, optimizer_seg, args)
            valid_perf = eval(model, device, valid_loader)
            test_perf = eval(model, device, test_loader)
            print('%3d\t%.6f\t%.6f\t%.6f'%(epoch, valid_perf['auc'], test_perf['auc'], loss))

            valid_curve.append(valid_perf['auc'])
            test_curve.append(test_perf['auc'])

            if no_improve_cnt > args.early_stop:
                break
            elif valid_perf['auc'] > best_valid_perf:
                best_valid_perf = valid_perf['auc']
                no_improve_cnt = 0
            else:
                no_improve_cnt += 1
        times.append((time.time() - t)/epoch)
        best_val_epoch = np.argmax(np.array(valid_curve))

        print('%2d-fold Valid\t%.6f'%(fold_idx+1, valid_curve[best_val_epoch]))
        print('%2d-fold Test\t%.6f'%(fold_idx+1, test_curve[best_val_epoch]))

        valid_list.append(valid_curve[best_val_epoch])
        test_list.append(test_curve[best_val_epoch])

 
    valid_list = np.array(valid_list)*100
    test_list = np.array(test_list)*100
    times = np.array(times)*1000
    print('Valid Acc:{:.4f}, Std:{:.4f}, Test Acc:{:.4f}, Std:{:.4f}'.format(np.mean(valid_list), np.std(valid_list), np.mean(test_list), np.std(test_list)))
    print('Mean time :{:.4f}, Std:{:.4f} (ms)'.format(np.mean(times), np.std(times)))
    with open('./ogbg.csv','a+')as f:
        f.write('{}\t{}\t{:.4f}\t{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
            args.read_op,args.gnn, args.drop, args.batch_size, args.head, args.relu, args.dataset, np.mean(valid_list), np.std(valid_list), np.mean(test_list), np.std(test_list))
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--gnn', type=str, default='gcn',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gcn)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    
    # SSRead parameters
    parser.add_argument('--read_op', type=str, default='weight_sum',
                        help='graph readout operation (default: sum)')
    parser.add_argument('--num_position', type=int, default=4,
                        help='number of structural positions (default: 4)')
    parser.add_argument('--gamma', type=float, default=0.01,
                        help='smoothing parameter for soft semantic alignment (default: 0.01)')

    # Optimization parameters
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--max_epochs', type=int, default=300,
                        help='maximum number of epochs to train (default: 500)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='initial learning rate of the optimizer (default: 0.001)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--early_stop', type=int, default=50,
                        help='patience for early stopping criterion (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducibility (default: 42)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    
    parser.add_argument('--norm', type=str, default='scale',
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--score_lin', type=str, default='1',
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--op', type=str, default='sum',
                        help='which gpu to use if any (default: 0)')
    
    # Dataset parameters
    parser.add_argument('--datapath', type=str, default="./dataset",
                        help='path to the directory of datasets (default: ./dataset)')
    parser.add_argument('--dataset', type=str, default="NCI1",
                        help='dataset name (default: NCI1)')
    parser.add_argument('--num_fold', type=int, default=10,
                        help='number of fold for cross-validation (default: 10)')
    
    parser.add_argument('--score_nums', type=int, default=9,
                        help='number of score nums')
    parser.add_argument('--head', type=int, default=4,
                        help='number of score nums')
    parser.add_argument('--drop', type=float, default=0.5,
                        help='number of score nums')
    parser.add_argument('--relu', type=str, default='relu',
                        help='number of score nums')
    parser.add_argument('--edge', type=str, default='',
                        help='number of score nums')
    parser.add_argument('--norm_x', type=str, default='',
                    help='number of score nums')
    parser.add_argument('--pos', type=str, default='pos',
                    help='number of score nums')
    
    args = parser.parse_args()
    datasets = ['NCI1','MUTAG','DD','PROTEINS','Mutagenicity']
    datasets = [
    'ogbg-moltox21',
    'ogbg-molbace',
    'ogbg-molbbbp',
    'ogbg-molclintox',
    'ogbg-molmuv',
    'ogbg-molsider',
    'ogbg-moltoxcast',
    'ogbg-molesol',
    'ogbg-molfreesolv',
    'ogbg-mollipo',
    'ogbg-molhiv'
    ]
    # for d in datasets:
    #     args.dataset = d
    #     try:
    main(args)
    # except:
    #     pass
