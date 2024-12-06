# ACRead Implementation

This repository contains the implementation of **ACRead**, a graph readout operation designed for graph classification tasks. Below are the required dependencies and model details.

## Dependencies

Ensure the following packages are installed before running the code:

```
torch=2.4.1
torch-cluster=1.6.3
torch-geometric=2.6.1
torch-scatter=2.1.2
torch-sparse=0.6.18
torch-spline-conv=1.2.2
tqdm=4.66.4
networkx=3.4.1
numpy=2.1.2
ogb=1.3.6
pandas=2.2.3
scipy=1.11.1
scikit-learn=1.3.0
networkx=2.8.8
```

## Model Parameters

The model configuration is as follows:

- **GNN Backbone:** Dropout rate of 0.5, 2 message-passing layers, 300-dimensional embeddings.
- **Optimization:** Batch size of 512, learning rate of 0.001, weight decay of 0.0001, early stopping with a patience of 50 epochs.
- **Evaluation:** Performed using 10-fold cross-validation.
- **Readout Operation:** ACRead readout operation using the Hadamard dot product as the $g_{\Psi}$ function, 9 centralities, 8 attention heads, $f_W$ is the different GNN backbone.


## Usage

### Graph Classification on the NCI1 Dataset

1. **Using GCN as the Backbone**  
   To run the graph classification task with GCN, execute the following command:

   ```bash
   python main.py --read_op acread --dataset NCI1 --gnn gcn --head 8 --batch_size 512 
   ```
