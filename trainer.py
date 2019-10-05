import os
import json
import time
from copy import deepcopy
import torch

from utils.common import save_model, get_bytes
from utils.logger import Logger
from utils.summary import EvaluationMetrics


class Trainer:
    def __init__(self, name, env, model, args=None):
        self.logger = Logger(name, verbose=True, args=args)
        path = os.path.join(self.logger.log_dir, 'config.json')
        with open(path, 'w') as f:
            json.dump(vars(args), f)

        self.train_loader = env['train']
        self.val_loader = env['test']
        self.num_classes = env['num_classes']
        self.model = model.to(args.device)
        self.args = args

        self.epoch = 0
        self.step = 0
        self.best_score = 0
        self.info = EvaluationMetrics(
            [
                'Epoch',
                'Layers',
                'Time/Step',
                'Time/Item',
                'Loss',
                'Accuracy/Top1',
                'Accuracy/Top5',
            ]
        )

        self.size = 0
        self.prune = False
        self.patience = args.patience

    def build(self):
        # Initialize weak learner statistics
        self.info.set(['γ'])
        if self.epoch == 0:
            self.model_curr = deepcopy(self.model)
            self.model_prev = deepcopy(self.model)
            self.alpha_prev = 0
            self.alpha_curr = -1
            self.gamma = -1
            self.gamma_tilde_prev = self.args.gamma_init
            self.gamma_tilde_curr = 0

            self.s = torch.zeros(
                len(self.train_loader),
                self.args.batch_size,
                self.num_classes
            ).to(self.args.device)
            self.update_cost()

        # Begin pruning if weak learning condition is satisfied
        elif self.gamma > self.args.gamma_thresh:
            self.prune = True
            self.patience = self.args.patience
            self.model_curr = deepcopy(self.model)

            self.model.prune(self.args.prune_rate)
            self.model = self.model.to(self.args.device)
            # Add layer if memory cannot be reduced
            if self.size == get_bytes(self.model):
                self.gamma = -1
                self.patience = 0
                return

            self.size = get_bytes(self.model)
            mbs = self.size/1024**2
            self.logger.log("pruning model to {0:.2f} MB...".format(mbs))

        # Add layer if out of patience
        elif self.prune:
            if not self.patience:
                self.model = self.model_curr
                self.size = get_bytes(self.model)
                mbs = self.size/1024**2
                self.logger.log("reverting model back to {0:.2f} MB...".format(mbs))

                self.logger.log("adding layer {}...".format(len(self.model) + 1))
                self.update_cost()
                self.model_prev = deepcopy(self.model)
                self.alpha_prev = deepcopy(self.alpha_curr)
                self.gamma_tilde_prev = deepcopy(self.gamma_tilde_curr)

                self.model.add()
                self.model = self.model.to(self.args.device)
                self.prune = False
                self.patience = self.args.patience
            else:
                self.patience -= 1

        gamma_tilde = 0
        self.model.train()
        for i, (data, labels) in enumerate(self.train_loader):
            self.step += 1
            st = time.time()

            data = data.to(self.args.device)
            labels = labels.to(self.args.device)
            outputs, loss = self.model.build(data, labels)
            gamma_tilde += torch.sum(outputs.detach()*self.C[i])

            elapsed = time.time() - st
            self.info.update('Epoch', self.epoch)
            self.info.update('Layers', len(self.model))
            self.info.update('Time/Step', elapsed)
            self.info.update('Time/Item', elapsed/self.args.batch_size)
            self.info.update('Loss', loss.item())
            self.info.update('γ', self.gamma)

            _, preds = torch.topk(outputs, 5)
            top1 = (labels == preds[:, 0]).float().mean()
            top5 = torch.sum((labels.unsqueeze(1).repeat(1, 5) == preds).float(), 1).mean()
            self.info.update('Accuracy/Top1', top1.item())
            self.info.update('Accuracy/Top5', top5.item())

            if self.step % self.args.log_step == 0:
                self.logger.log("logging values at step: {}".format(self.step))
                self.logger.scalar_summary(self.info.avg, self.step, filename='train.csv')
                self.info.reset()

        self.epoch += 1
        gamma_tilde /= -self.Z
        self.alpha_curr = 0.5*torch.log((1 + gamma_tilde)/(1 - gamma_tilde))
        self.gamma_tilde_curr = gamma_tilde

        gamma = (gamma_tilde**2 - self.gamma_tilde_prev**2)
        gamma /= (1 - self.gamma_tilde_prev**2)
        gamma = torch.sign(gamma)*torch.sqrt(torch.abs(gamma))
        self.gamma = gamma.item()

    def finetune(self):
        self.model.train()
        for data, labels in self.train_loader:
            self.step += 1
            st = time.time()

            data = data.to(self.args.device)
            labels = labels.to(self.args.device)
            outputs, loss = self.model.finetune(data, labels)

            elapsed = time.time() - st
            self.info.update('Epoch', self.epoch)
            self.info.update('Layers', len(self.model))
            self.info.update('Time/Step', elapsed)
            self.info.update('Time/Item', elapsed/self.args.batch_size)
            self.info.update('Loss', loss.item())

            _, preds = torch.topk(outputs, 5)
            top1 = (labels == preds[:, 0]).float().mean()
            top5 = torch.sum((labels.unsqueeze(1).repeat(1, 5) == preds).float(), 1).mean()
            self.info.update('Accuracy/Top1', top1.item())
            self.info.update('Accuracy/Top5', top5.item())

            if self.step % self.args.log_step == 0:
                self.logger.log("logging values at step: {}".format(self.step))
                self.logger.scalar_summary(self.info.avg, self.step, filename='train.csv')
                self.info.reset()

        self.epoch += 1

    def infer(self):
        info = EvaluationMetrics(
            [
                'Epoch',
                'Layers',
                'Time/Total',
                'Time/Item',
                'Accuracy/Top1',
                'Accuracy/Top5'
            ]
        )
        self.model.eval()
        self.logger.log("evaluating values at step: {}".format(self.step))
        with torch.no_grad():
            st = time.time()
            for data, labels in self.logger.progress(self.val_loader):
                b_st = time.time()

                data = data.to(self.args.device)
                labels = labels.to(self.args.device)

                outputs = self.model(data)
                _, preds = torch.topk(outputs, 5)
                top1 = (labels == preds[:, 0]).float().mean()
                top5 = torch.sum((labels.unsqueeze(1).repeat(1, 5) == preds).float(), 1).mean()
                info.update('Accuracy/Top1', top1.item())
                info.update('Accuracy/Top5', top5.item())

                elapsed = time.time() - b_st
                info.update('Time/Item', elapsed/len(data))

        elapsed = time.time() - st
        info.update('Epoch', self.epoch)
        info.update('Layers', len(self.model))
        info.update('Time/Total', elapsed)

        self.logger.scalar_summary(info.avg, self.step, filename='infer.csv')
        score = info.avg['Accuracy/Top1']
        if score > self.best_score:
            self.logger.log("updating model checkpoint...")
            self.best_score = score

            path = os.path.join(self.logger.log_dir, "model.pth")
            save_model(self.model, path)

    def update_cost(self):
        self.logger.log("updating cost matrix...")
        self.C = torch.zeros_like(self.s)
        self.Z = 0
        data = self.logger.progress(self.train_loader)

        self.model.eval()
        self.model_prev.eval()
        for i, (data, labels) in enumerate(data):
            data = data.to(self.args.device)
            labels = labels.to(self.args.device)

            if self.epoch > 0:
                outputs = self.model(data).detach()
                outputs_prev = self.model_prev(data).detach()
                self.s[i] += self.alpha_curr*outputs - self.alpha_prev*outputs_prev

            for j, label in enumerate(labels):
                s = self.s[i][j]
                local_sum = 0
                for l in range(self.num_classes):
                    if l != label:
                        self.C[i][j][l] = torch.exp(s[l] - s[label])
                        local_sum += self.C[i][j][l]
                self.C[i][j][label] = -local_sum
                self.Z += local_sum
