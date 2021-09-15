import math
import torch
import torch.nn.functional as F

class DynamicPool3d(torch.nn.Module):
    def __init__(self, scale, steps, mode):
        super().__init__()

        self.scale = scale #in physical units
        self.steps = steps
        self.mode = mode
        self.kernel_size = tuple([math.floor(self.scale/step) if step < self.scale else 1 for step in self.steps])

    def forward(self, input):

        #kernel_size = []
        #output_steps = []
        #for step in self.steps:
        #    if step < self.scale:
        #        kernel_dim = math.floor(self.scale/step)
        #        kernel_size.append(kernel_dim)
        #        output_steps.append(kernel_dim*step)
        #    else:
        #        kernel_size.append(1)
        #        output_steps.append(step)


        if self.mode == 'maxpool3d':
            out = F.max_pool3d(input, self.kernel_size, stride=self.kernel_size)
        elif self.mode == 'average':
            out = F.avg_pool3d(input, self.kernel_size, stride=self.kernel_size)

        return out
