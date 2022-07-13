import math
import torch
import torch.nn.functional as F

class DynamicPool3d(torch.nn.Module):
    def __init__(self, scale, steps, mode, irreps):
        super().__init__()

        self.scale = scale #in physical units
        self.steps = steps
        self.mode = mode
        self.kernel_size = tuple([math.floor(self.scale/step) if step < self.scale else 1 for step in self.steps])
        self.irreps = irreps

    def forward(self, input):

        if self.mode == 'maxpool3d':
        
            out = max_pool3d(input, self.irreps, self.kernel_size, stride=self.kernel_size) #e3nn max_pool3d implementation
            #out = F.max_pool3d(input, self.kernel_size, stride=self.kernel_size) #non-equivariant pytorch implementation
        elif self.mode == 'average':
            out = F.avg_pool3d(input, self.kernel_size, stride=self.kernel_size)

        return out

def max_pool3d(input, irreps, kernel_size, stride):

    assert input.shape[1] == irreps.dim, "Shape mismatch"
    cat_list = []

    start = 0
    for i in irreps.ls:

        end = start + 2*i+1
        temp = input[:,start:end,...]
        if i == 0:
            pooled,indices = F.max_pool3d_with_indices(temp[:,0,...],kernel_size,stride=stride,return_indices=True)
            cat_list.append(pooled)
        else:
            pooled, indices = F.max_pool3d_with_indices(temp.norm(dim = 1),kernel_size,stride=stride,return_indices=True)
            for slice in range(2*i+1):
                pooled = temp[:,slice,...].flatten()[indices]
                cat_list.append(pooled)
        start = end

    return torch.stack(tuple(cat_list),dim = 1)

def max_pool3d_optimized(input, irreps, kernel_size, stride):

    assert input.shape[1] == irreps.dim, "Shape mismatch"
    cat_list = []

    start = 0
    for i in irreps.ls:
        # x = [batch, mul * dim, x, y, z]
        # x = [batch, mul, dim, x, y, z]
        # norm = [batch, mul, x, y, z]
        # indices = [batch, mul, x, z, y]
        # x = x.transpose(0, 2)  [dim, batch, mul, x, y, z]
        # x[:, indices]  [dim, batch, mul, x, y, z]
        #
        # x.transpose(0, 1).flatten(1)[:, i].transpose(0, 1).shape
        # [6]: _, i = torch.nn.functional.max_pool2d_with_indices(x.pow(2).sum(1), 2, stride=2, return_indices=True)

        end = start + 2*i+1
        temp = input[:,start:end,...]
        if i == 0:
            pooled,indices = F.max_pool3d_with_indices(temp[:,0,...],kernel_size,stride=stride,return_indices=True)
            cat_list.append(pooled)
        else:
            pooled, indices = F.max_pool3d_with_indices(temp.norm(dim = 1),kernel_size,stride=stride,return_indices=True)
            for slice in range(2*i+1):
                pooled = temp[:,slice,...].flatten()[indices]
                cat_list.append(pooled)
        start = end

    return torch.stack(tuple(cat_list),dim = 1)
def max_pool3d(input, irreps, kernel_size, stride):

    assert input.shape[1] == irreps.dim, "Shape mismatch"
    cat_list = []

    start = 0
    for i in irreps.ls:
        # x = [batch, mul * dim, x, y, z]
        # x = [batch, mul, dim, x, y, z]
        # norm = [batch, mul, x, y, z]
        # indices = [batch, mul, x, z, y]
        # x = x.transpose(0, 2)  [dim, batch, mul, x, y, z]
        # x[:, indices]  [dim, batch, mul, x, y, z]
        #
        # x.transpose(0, 1).flatten(1)[:, i].transpose(0, 1).shape
        # [6]: _, i = torch.nn.functional.max_pool2d_with_indices(x.pow(2).sum(1), 2, stride=2, return_indices=True)

        end = start + 2*i+1
        temp = input[:,start:end,...]
        if i == 0:
            pooled,indices = F.max_pool3d_with_indices(temp[:,0,...],kernel_size,stride=stride,return_indices=True)
            cat_list.append(pooled)
        else:
            pooled, indices = F.max_pool3d_with_indices(temp.norm(dim = 1),kernel_size,stride=stride,return_indices=True)
            for slice in range(2*i+1):
                pooled = temp[:,slice,...].flatten()[indices]
                cat_list.append(pooled)
        start = end

    return torch.stack(tuple(cat_list),dim = 1)