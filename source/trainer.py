import torch
import torch.nn as nn
from torch.optim import Adam
import time
import os

class Trainer:

    def __init__(self, option, model, train_dataloader, test_dataloader=None):

        # Superparam
        lr = option.lr
        betas = (option.adam_beta1, option.adam_beta2)
        weight_decay = option.adam_weight_decay
        with_cuda = option.with_cuda
        cuda_devices = option.cuda_devices
        self.log_freq = option.log_freq
        self.save_path = option.this_expsdir
        self.epochs = option.epochs
        self.clip_norm = option.clip_norm

        self.best_test_acc = 0

        self.start = time.time()
        self.msg_with_time = lambda msg: \
                "%s Time elapsed %0.2f hrs (%0.1f mins)" \
                % (msg, (time.time() - self.start) / 3600.,
                        (time.time() - self.start) / 60.)


        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:{}".format(cuda_devices[0]) if cuda_condition else "cpu")


        self.model = model.to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1 and len(cuda_devices) > 1:
            print("Using %d GPUS for train" % len(cuda_devices))
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss()
        #self.criterion = nn.CrossEntropyLoss()

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self):
        for epoch in range(self.epochs):
            train_acc = self.iteration(epoch,self.train_data)
            if self.test_data is not None:
                test_acc = self.iteration(epoch, self.test_data,train=False)
                self.best_test_acc =  max(self.best_test_acc,test_acc)

                if self.best_test_acc == test_acc:
                    self.save_model(epoch)

    def test(self, epoch):
        test_acc = self.iteration(epoch, self.test_data, train=False)
        self.best_test_acc = max(self.best_test_acc,test_acc)


    def iteration(self, epoch, data_loader, train=True):

        str_code = "train" if train else "test"

        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        count = 0

        if train:
            self.model.train()
            for i, data in enumerate(data_loader):
                data = {key: value.to(self.device) for key, value in data.items()}
                #print(data['x'][3:5,0,0])

                output = self.model.forward(data['x'],data['adj'])
                # print(output)
                loss = self.criterion(output, data["label"])

                self.optim.zero_grad()
                loss.backward()
                if self.clip_norm>0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clip_norm)
                self.optim.step()

                avg_loss += loss.item()
                # print(output)
                # print(output.argmax(dim=-1))
                # print(data['label'])

                correct = output.argmax(dim=-1).eq(data['label']).sum().item()
                # print(correct)
                # print('*' * 100)
                acc = correct / data['label'].nelement()
                total_correct += correct
                total_element += data['label'].nelement()
                count+=1

                msg = 'epoch:%d, iter:%d, loss:%0.3f, acc:%0.2f, avg_acc:%0.2f' % (epoch, i, loss.item(),acc*100,
                                                                             total_correct/total_element*100)
        else:
            self.model.eval()
            with torch.no_grad():
                for i, data in enumerate(data_loader):

                    data = {key: value.to(self.device) for key, value in data.items()}
                    output = self.model.forward(data['x'], data['adj'])
                    loss = self.criterion(output, data["label"])
                    avg_loss += loss.item()
                    # print('*'*20+str(i)+'*'*20)
                    # print('input：',torch.unique(data['x']))
                    # print('out：',output.argmax(dim=-1))
                    # print('label：',data['label'])

                    correct = output.argmax(dim=-1).eq(data['label']).sum().item()
                    acc = correct / data['label'].nelement()
                    total_correct += correct
                    total_element += data['label'].nelement()
                    count+=1
            self.model.train()

        msg ='Epoch:%d, %s, avg_loss:%0.2f, avg_acc:%0.2f'%(epoch, str_code, avg_loss/count,
                                                      total_correct/total_element*100)
        log = self.msg_with_time(msg)
        print(log)
        self.save_log(log)
        return total_correct/total_element*100

    def save_model(self, epoch):
        with open(os.path.join(self.save_path,'best_model.pkl'), 'wb') as f:
            torch.save(self.model.state_dict(), f)
        log = "Epoch:%d Model Saved" % epoch
        print(log)
        self.save_log(log)

    def save_log(self,strs):
        with open(os.path.join(self.save_path,'log.txt'),'a+') as f:
            f.write(strs)
            f.write('\n')





