from copy import deepcopy

import torch
from torch import nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

def get_named_child(module, name):
    stack = [int(tk) if tk.isnumeric() else tk for tk in name.split('.')]
    for depth in stack:
        if isinstance(depth, str):
            module = getattr(module, depth)
        else:
            try:
                module = module[depth]
            except KeyError:
                module = module[str(depth)]
    return module

def set_named_child(module, name, newmodule):
    stack = [int(tk) if tk.isnumeric() else tk for tk in name.split('.')]
    for i, depth in enumerate(stack):
        if i < len(stack)-1:
            if isinstance(depth, str):
                module = getattr(module, depth)
            else:
                try:
                    module = module[depth]
                except KeyError:
                    module = module[str(depth)]
        else:
            if isinstance(depth, str):
                setattr(module, depth, newmodule)
            else:
                try:
                    _ = module[depth]
                    module[depth] = newmodule
                except KeyError:
                    module[str(depth)] = newmodule

class MultiBN(nn.Module):
    def __init__(self, bn):
        super().__init__()
        self.current_index = 0

        self.bn1 = deepcopy(bn)
        self.bn2 = deepcopy(bn)
        self.bn3 = deepcopy(bn)

    def forward(self, x, index=None):
        if index is None:
            index = self.current_index

        if index == 0:
            return self.bn1(x)
        elif index == 1:
            return self.bn2(x)
        elif index == 2:
            return self.bn3(x)
        else:
            raise ValueError(f"Invalid index {index}. Must be 0, 1, or 2.")

class MultiBN1d(nn.Module):
    def __init__(self, bn):
        super().__init__()
        self.current_index = 0

        self.bn1 = deepcopy(bn)
        self.bn2 = deepcopy(bn)
        self.bn3 = deepcopy(bn)

    def forward(self, x, index=None):
        if index is None:
            index = self.current_index

        if index == 0:
            return self.bn1(x)
        elif index == 1:
            return self.bn2(x)
        elif index == 2:
            return self.bn3(x)
        else:
            raise ValueError(f"Invalid index {index}. Must be 0, 1, or 2.")

class MultiBNModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.current_index = 0
        self.bns = [k for k,v in self.model.named_modules() if isinstance(v, (nn.BatchNorm2d, nn.BatchNorm1d))]
        # override batchnorms
        for bn in self.bns:
            module = get_named_child(self.model, bn)
            if isinstance(module, nn.BatchNorm2d):
                multi_bn = MultiBN(module)
                multi_bn.bn1.load_state_dict(module.state_dict())
                multi_bn.bn2.load_state_dict(module.state_dict())
                multi_bn.bn3.load_state_dict(module.state_dict())
                set_named_child(self.model, bn, multi_bn)
            elif isinstance(module, nn.BatchNorm1d):
                multi_bn = MultiBN1d(module)
                multi_bn.bn1.load_state_dict(module.state_dict())
                multi_bn.bn2.load_state_dict(module.state_dict())
                multi_bn.bn3.load_state_dict(module.state_dict())
                set_named_child(self.model, bn, multi_bn)

    @torch.no_grad()
    def update_index(self, index=0):
        self.current_index = index
        for bn in self.bns:
            m = get_named_child(self.model, bn)
            m.current_index = index

    @torch.no_grad()
    def clean_param(self, p):
        p[torch.isnan(p)] = 0
        p[torch.isinf(p)] = 0
        return p

    @torch.no_grad()
    def ema_alternate(self, rate=.85):
        for bn in self.bns:
            m = get_named_child(self.model, bn)
            for p1, p2, p3 in zip(m.bn1.parameters(), m.bn2.parameters(), m.bn3.parameters()):
                p2.data = rate * p1.data + (1 - rate) * p2.data
                p3.data = rate * p1.data + (1 - rate) * p3.data

                # t = .5*self.clean_param(p1.data) + .5*self.clean_param(p2.data)
                # p1.data = self.clean_param(rate*t + (1-rate)*p1.data)
                # p2.data = self.clean_param(rate*t + (1-rate)*p2.data)


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

class LateFuse(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.rgb = deeplabv3_mobilenet_v3_large(num_classes=num_classes)
        self.dth = deeplabv3_mobilenet_v3_large(num_classes=num_classes)
        with torch.no_grad():
            Co, _, K1, K2 = self.dth.backbone['0'][0].weight.shape
            w = torch.empty(Co, 1, K1, K2)
            torch.nn.init.xavier_uniform_(w)
            self.dth.backbone['0'][0].weight = torch.nn.Parameter(w)
        self.merge = nn.Conv2d(2*num_classes, num_classes, 1, bias=False)

    def forward(self, c, d):
        c = self.rgb(c)['out']
        d = self.dth(d)['out']
        x = torch.cat([c,d], dim=1)
        x = self.merge(x)
        return {'out': x}

class EarlyFuse(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dlv3 = deeplabv3_mobilenet_v3_large(num_classes=num_classes)
        with torch.no_grad():
            Co, _, K1, K2 = self.dlv3.backbone['0'][0].weight.shape
            w = torch.empty(Co, 4, K1, K2)
            nn.init.xavier_uniform_(w)
            self.dlv3.backbone['0'][0].weight = nn.Parameter(w)

    def forward(self, x):
        return self.dlv3(x)

if __name__ == "__main__":
    EarlyFuse(28)
