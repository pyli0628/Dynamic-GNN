import argparse
import os
import torch
from torch.utils.data import DataLoader
from model import *
from trainer import *
from data import *
import random
import numpy as np
import time


class Option(object):
    def __init__(self, d):
        self.__dict__ = d

    def save(self):
        with open(os.path.join(self.this_expsdir, "option.txt"), "w") as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                f.write("%s, %s\n" % (key, str(value)))


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", required=True, type=str, help="train dataset for train")
    parser.add_argument("--test_data", type=str, default=None, help="test set for evaluate")
    parser.add_argument("--train_label", required=True, type=str, help="train dataset label")
    parser.add_argument("--test_label", type=str, default=None, help="test set label")
    parser.add_argument("--output_path", required=True, type=str, help="out/")
    parser.add_argument('--exp_name', default='test', type=str)

    parser.add_argument('--model', type=int, default=0)
    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument('--layer_norm', default=False, action="store_true")
    parser.add_argument("--infeature", type=int, default=1, help="the input feature dim")
    parser.add_argument("--hidden", type=int, default=8, help="hidden size of transformer model")
    parser.add_argument("--layers", type=int, default=3, help="number of layers")
    parser.add_argument("--heads", type=int, default=4, help="number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clip_norm", type=float, default=0.0)


    parser.add_argument("--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.0005, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--eval', default=False, action="store_true")

    d = vars(parser.parse_args())
    args = Option(d)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.exp_name is None:
      args.tag = time.strftime("%y-%m-%d-%H-%M")
    else:
      args.tag = args.exp_name
    args.this_expsdir = os.path.join(args.output_path, args.tag)
    if not os.path.exists(args.this_expsdir):
        os.makedirs(args.this_expsdir)

    args.save()
    print("Option saved.")


    print("Loading Train Dataset", args.train_data)
    train_dataset = Data(args.train_data, args.train_label, args.infeature)

    print("Loading Test Dataset", args.test_data)
    test_dataset = Data(args.test_data, args.test_label,args.infeature,train=False) \
        if args.test_data is not None else None




    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    print("Building model")

    if args.model==0:
        print("Building dynamic model")
        model = DG_fc(args.infeature,args.hidden,args.heads,args.layers,args.dropout,args.layer_norm)
        print("Creating Trainer")
        trainer = Trainer(args, model, train_dataloader=train_data_loader, test_dataloader=test_data_loader)
    elif args.model==1:
        print("Building gat model")
        model = GAT(args.infeature,args.hidden,args.heads,args.layers,args.dropout)
        print('Creating Trainer')
        trainer = Trainer(args, model, train_dataloader=train_data_loader, test_dataloader=test_data_loader)

    if args.load is  not None:
        with open(args.load, 'rb') as f:
            model.load_state_dict(torch.load(f))
        if args.eval:
            model.eval()


    print("Training Start")
    trainer.train()

if __name__ == '__main__':
    train()
