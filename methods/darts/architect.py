from copy import deepcopy
import torch
import torch.nn as nn


class Architect:
    def __init__(self, controller, hessian=False, args=None):
        self.net = controller
        self.v_net = deepcopy(controller)
        self.criterion = nn.CrossEntropyLoss()
        self.hessian = hessian
        self.args = args

    def virtual_step(self, X_train, Y_train, optimizer):
        outputs = self.net(X_train)
        loss = self.criterion(outputs, Y_train)
        grads = torch.autograd.grad(loss, self.net.weights)

        with torch.no_grad():
            for w, vw, g in zip(self.net.weights, self.v_net.weights, grads):
                m = optimizer.state[w].get('momentum_buffer', 0)
                m *= self.args.momentum
                vw.copy_(w - self.args.lr * (m + g + self.args.weight_decay*w))

            for a, va in zip(self.net.alphas, self.v_net.alphas):
                va.copy_(a)

    def backward(self, X_train, Y_train, X_val, Y_val, optimizer):
        outputs = self.v_net(X_val)
        loss = self.criterion(outputs, Y_val)
        v_alphas = tuple(self.v_net.alphas)
        v_weights = tuple(self.v_net.weights)
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)

        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]
        if self.hessian:
            hessian = self.compute_hessian(dw, X_train, Y_train)
            with torch.no_grad():
                for a, da, h in zip(self.net.alphas, dalpha, hessian):
                    a.grad = da - self.args.lr*h
        else:
            with torch.no_grad():
                for a, da in zip(self.net.alphas, dalpha):
                    a.grad = da

    def compute_hessian(self, dw, X_train, Y_train):
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01/norm

        with torch.no_grad():
            for p, d in zip(self.net.weights, dw):
                p += eps*d
        outputs = self.net(X_train)
        loss = self.criterion(outputs, Y_train)
        dalpha_pos = torch.autograd.grad(loss, self.net.alphas)

        with torch.no_grad():
            for p, d in zip(self.net.weights, dw):
                p -= 2*eps*d
        outputs = self.net(X_train)
        loss = self.criterion(outputs, Y_train)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas)

        with torch.no_grad():
            for p, d in zip(self.net.weights, dw):
                p += eps*d

        return [(p - n)/2*eps for p, n in zip(dalpha_pos, dalpha_neg)]
