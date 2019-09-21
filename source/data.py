from torch.utils.data import Dataset
import torch
import random
import numpy as np
import scipy.io as sio
import pandas as pd
class Data(Dataset):
    def __init__(self, data_path,label_path, infeature, train=True):

        # load data
        self.data = self.load_mat(data_path,train=train)
        self.label = self.load_label(label_path,train=train)
        assert len(self.data)==len(self.label)
        self.infeature = infeature
        print('dataset shape:',self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, adj= self.gen_graph(idx)
        label = self.label[idx]
        x, adj, label = torch.FloatTensor(x),torch.FloatTensor(adj),torch.tensor(label).long()
        output = {'x':x,'adj':adj,'label':label}
        return output

    def gen_graph(self,idx):
        x = self.data[idx] #(n,L)
        N,L = x.shape
        if self.infeature==1:
            x = np.expand_dims(x,axis=2) #(n,L,1)
        elif self.infeature==L:
            x = x
        else:
            x = x.reshape(x.shape[0], -1, self.infeature)
        adj = np.ones((len(x),len(x)))
        return x,adj

    def load_mat(self,path, train=True):
        data = sio.loadmat(path)
        key = 'data_train' if train else 'data_val'
        return data[key] # if train else np.zeros((1384,62,200))
    def load_label(self,path,train=True):
        data = sio.loadmat(path)
        key = 'Y_train' if train else 'Y_val'
        label = data[key].squeeze()
        label = np.where(label>=0,label,2)
        return label



        # mol = MolFromSmiles(smi)
        # adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
        # feats = []  #7
        # for atom in mol.GetAtoms():
        #     feat = []
        #     feat.append(atom.GetAtomicNum())
        #     feat.append(atom.GetDegree())
        #     feat.append(atom.GetTotalNumHs())
        #     feat.append(atom.GetImplicitValence())
        #     feat.append(atom.GetIsAromatic())
        #     feat.append(len(atom.GetNeighbors()))
        #     feat.append(atom.IsInRing())
        #
        #     feats.append(feat)
        #

        # #cut and padding
        # adj = adj[:self.max_length,:self.max_length]
        # adj_pad = np.zeros((self.max_length,self.max_length))
        # adj_pad[:len(adj),:len(adj)] = adj + np.eye(len(adj))
        #
        # feats = feats[:self.max_length]
        # padding = [[0]*7 for _ in range(self.max_length-len(feats))]
        # feats.extend(padding)
        # feats = np.array(feats)
        #
        # return feats,adj_pad