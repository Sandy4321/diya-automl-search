import os
import time
import torch
import envs
from trainer import Trainer
from methods.base import Base
from methods.darts.controller import CNNController
from methods.darts.controller import RNNController
from methods.darts.controller import TransformerController
from methods.darts.architect import Architect


class DARTSTrainer(Trainer):
    def __init__(self, env, controller, hessian,
                 logger=None, args=None):
        super().__init__(env, controller, args)
        self.optimizer = torch.optim.SGD(
            controller.weights,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        self.alpha_optimizer = torch.optim.Adam(
            controller.alphas,
            lr=args.lr*0.01,
            betas=(0.5, 0.999),
            weight_decay=args.weight_decay*10
        )
        self.architect = Architect(controller, hessian, args)
        self.logger = logger

    def train(self):
        self.model.train()
        for data, labels in self.env['train']:
            self.step += 1
            st = time.time()

            data = data.to(self.args.device)
            labels = labels.to(self.args.device)

            v_data, v_labels = next(iter(self.env['val']))
            v_data = v_data.to(self.args.device)
            v_labels = v_labels.to(self.args.device)

            self.alpha_optimizer.zero_grad()
            self.architect.backward(data, labels, v_data, v_labels, self.optimizer)
            self.alpha_optimizer.step()

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

            if self.step % self.args.log_step == 0:
                self.logger.log("Training statistics for step: {}".format(self.step))
                self.logger.scalar_summary(self.info.avg, self.step)
                self.info.reset()

                print(self.model.alpha_ops[0])

        self.epoch += 1


class DARTS(Base):
    def __init__(self, args):
        super().__init__('DARTS', args)
        if args.type == 'cnn':
            controller = CNNController
        elif args.type == 'rnn':
            controller = RNNController
        elif args.type == 'transformer':
            controller = TransformerController
        else:
            raise NotImplementedError
        self.env = getattr(envs, args.env)(args)
        self.controller = controller(
            self.env['size'],
            self.env['num_classes'],
            args
        )
        self.trainer = DARTSTrainer(
            self.env,
            self.controller,
            hessian=False,
            logger=self.logger,
            args=args
        )
        self.args = args

    def search(self):
        best_acc = 0
        for epoch in range(self.args.epochs):
            self.trainer.train()
            self.trainer.infer(test=True)
            acc = self.trainer.info.avg['Accuracy/Top1']
            self.trainer.info.reset()
            self.logger.log("Validation accuracy: {}".format(acc))
            if acc > best_acc:
                best_acc = acc
                path = os.path.join(self.logger.log_dir, 'model.pth'.format(epoch))
                self.logger.log("Saving model at epoch: {}".format(epoch))


class DARTS_2ND(DARTS):
    def __init__(self, args):
        super().__init__(args)
        self.trainer = DARTSTrainer(
            self.env,
            self.controller,
            hessian=True,
            logger=self.logger,
            args=args
        )
