import time
import torch
import torch.nn as nn

from utils.summary import EvaluationMetrics


class Trainer:
    def __init__(self, env, model, args=None):
        self.loader = env
        self.model = model.to(args.device)
        self.args = args

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum
        )
        self.epoch = 0
        self.step = 0
        self.info = EvaluationMetrics(
            [
                'Epoch',
                'Time/Total',
                'Time/Step',
                'Time/Item',
                'Loss',
                'Accuracy/Top1',
            ]
        )

    def train(self):
        self.model.train()
        for data, labels in self.loader:
            self.step += 1
            st = time.time()

            data = data.to(self.args.device)
            labels = labels.to(self.args.device)
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            elapsed = time.time() - st
            self.info.update('Epoch', self.epoch)
            self.info.update('Time/Step', elapsed)
            self.info.update('Time/Item', elapsed/self.args.batch_size)
            self.info.update('Loss', loss.item())

            _, preds = torch.max(outputs, dim=-1)
            top1 = (labels == preds).float().mean()
            self.info.update('Accuracy/Top1', top1.item())

        self.epoch += 1

    def infer(self):
        self.info.reset()
        self.model.eval()
        with torch.no_grad():
            st = time.time()
            for data, labels in self.loader:
                b_st = time.time()

                data = data.to(self.args.device)
                labels = labels.to(self.args.device)

                outputs = self.model(data)
                _, preds = torch.max(outputs, dim=-1)
                top1 = (labels == preds).float().mean()
                self.info.update('Accuracy/Top1', top1.item())

                elapsed = time.time() - b_st
                self.info.update('Time/Item', elapsed/len(data))

        elapsed = time.time() - st
        self.info.update('Epoch', self.epoch)
        self.info.update('Time/Total', elapsed)
