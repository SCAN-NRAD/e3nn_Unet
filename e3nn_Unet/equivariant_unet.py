from functools import partial
import torch
from torch import nn
from e3nn.nn import BatchNorm, Gate, Dropout
from e3nn.nn.models.v2103.voxel_convolution import LowPassFilter
from e3nn.o3 import Irreps, Linear
from nnunet.network_architecture.neural_network import SegmentationNetwork
from .voxel_convolution import ConvolutionVoxel as Convolution

#Deprecated
class ConvolutionBlock(nn.Module):
    def __init__(self, input, irreps_hidden, activation, irreps_sh, normalization,kernel_size,dropout_prob,cutoff):
        super().__init__()

        if normalization == 'None':
            BN = Identity
        elif normalization == 'batch':
            BN = BatchNorm
        elif normalization == 'instance':
            BN = partial(BatchNorm,instance=True)


        irreps_scalars = Irreps( [ (mul, ir) for mul, ir in irreps_hidden if ir.l == 0 ] )
        irreps_gated   = Irreps( [ (mul, ir) for mul, ir in irreps_hidden if ir.l > 0  ] )
        fe = sum(mul for mul,ir in irreps_gated if ir.p == 1)
        fo = sum(mul for mul,ir in irreps_gated if ir.p == -1)
        irreps_gates = Irreps(f"{fe}x0e+{fo}x0o").simplify()

        if irreps_gates.dim == 0:
            irreps_gates = irreps_gates.simplify()
            activation_gate = []
        else:
            activation_gate = [torch.sigmoid, torch.tanh][:len(activation)]

        self.gate1 = Gate(irreps_scalars, activation, irreps_gates, activation_gate, irreps_gated)
        self.conv1 = Convolution(input, self.gate1.irreps_in, irreps_sh, kernel_size,cutoff=cutoff)
        self.batchnorm1 = BN(self.gate1.irreps_in)
        self.dropout1 = Dropout(self.gate1.irreps_out, dropout_prob)

        self.gate2 = Gate(irreps_scalars, activation, irreps_gates, activation_gate, irreps_gated)
        self.conv2 = Convolution(self.gate1.irreps_out, self.gate2.irreps_in, irreps_sh, kernel_size,cutoff=cutoff)
        self.batchnorm2 = BN(self.gate2.irreps_in)
        self.dropout2 = Dropout(self.gate2.irreps_out, dropout_prob)

        self.irreps_out = self.gate2.irreps_out

    def forward(self, x):
 
        x = self.conv1(x)
        x = self.batchnorm1(x.transpose(1, 4)).transpose(1, 4)
        x = self.gate1(x.transpose(1, 4)).transpose(1, 4)
        x = self.dropout1(x.transpose(1, 4)).transpose(1, 4)

        x = self.conv2(x)
        x = self.batchnorm2(x.transpose(1, 4)).transpose(1, 4)
        x = self.gate2(x.transpose(1, 4)).transpose(1, 4)
        x = self.dropout2(x.transpose(1, 4)).transpose(1, 4)
        return x

class Down(nn.Module):
    def __init__(self, n_blocks_down,activation,irreps_sh,ne,no,BN,input,kernel_size,down_op,scale,stride,dropout_prob,cutoff):
        super().__init__()

        blocks = []
        self.down_irreps_out = []

        for n in range(n_blocks_down+1):
            irreps_hidden = Irreps(f"{4*ne}x0e + {4*no}x0o + {2*ne}x1e + {ne}x2e + {2*no}x1o + {no}x2o").simplify()
            block = ConvolutionBlock(input,irreps_hidden,activation,irreps_sh,BN,kernel_size,dropout_prob,cutoff)
            blocks.append(block)
            self.down_irreps_out.append(block.irreps_out)
            input = block.irreps_out
            ne *= 2
            no *= 2

        self.down_blocks = nn.ModuleList(blocks)

        #change to pooling
        if down_op == 'lowpass':
            self.pool       = LowPassFilter(scale,stride=stride)
        elif down_op == 'maxpool3d':
            self.pool       = nn.MaxPool3d(scale,stride=stride)
        elif down_op == 'average':
            self.pool       = nn.AvgPool3d(scale,stride=stride)

    def forward(self, x):
        ftrs = []
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            ftrs.append(x)
            if i < len(self.down_blocks)-1:
                x = self.pool(x)
        return ftrs

class Up(nn.Module):
    def __init__(self, n_blocks_up,activation,irreps_sh,ne,no,BN,downblock_irreps,kernel_size,up_op,scale,stride,dropout_prob,scalar_upsampling,cutoff):
        super().__init__()

        self.n_blocks_up = n_blocks_up
        if up_op == 'lowpass':
            self.upsamp    = LowPassFilter(scale,stride=stride,transposed=True)
        else:
            self.upsamp = nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=True)

        input = downblock_irreps[-1]
        blocks = []

        for n in range(n_blocks_up):
            if scalar_upsampling:
                irreps_hidden = Irreps(f"{8*ne}x0e+{8*no}x0o").simplify()
            else:
                irreps_hidden = Irreps(f"{4*ne}x0e + {4*no}x0o + {2*ne}x1e + {ne}x2e + {2*no}x1o + {no}x2o").simplify()

            block = ConvolutionBlock(input+downblock_irreps[::-1][n+1],irreps_hidden,activation,irreps_sh,BN,kernel_size,dropout_prob,cutoff)
            blocks.append(block)
            input = block.irreps_out
            ne //= 2
            no //= 2

        self.up_blocks = nn.ModuleList(blocks)

    def forward(self, x, down_features):

        for i in range(self.n_blocks_up):
            x        = self.upsamp(x)
            x        = torch.cat([x, down_features[::-1][i+1]], dim=1)
            x        = self.up_blocks[i](x)
        return x

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def forward(self, x):
        return x

class UNet(SegmentationNetwork):
    def __init__(self, n_classes_scalar, n_classes_vector, batch_norm='instance', n=2, n_downsample = 2, equivariance = 'SO3',
        lmax = 2, kernel_size = 5, down_op = 'lowpass', stride = 2, scale =2,input_irreps="0e",
        is_bias = True,scalar_upsampling=False,dropout_prob=0,cutoff=False):
        super().__init__()

        self.n_classes_scalar = n_classes_scalar
        self.num_classes = n_classes_scalar + 3*n_classes_vector
        output_irreps = Irreps(f"{n_classes_scalar}x0e + {n_classes_vector}x1e")

        self.n_downsample = n_downsample



        assert batch_norm in ['None','batch','instance'], "batch_norm needs to be 'batch', 'instance', or 'None'"
        assert down_op in ['maxpool3d','average','lowpass'], "down_op needs to be 'maxpool3d', 'average', or 'lowpass'"

        if down_op == 'lowpass':
            up_op = 'lowpass'
            self.odd_resize = True

        else:
            up_op = 'upsample'
            self.odd_resize = False

        if equivariance == 'SO3':
            activation = [torch.relu]
            irreps_sh = Irreps.spherical_harmonics(lmax, 1)
            ne = n
            no = 0
        elif equivariance == 'O3':
            activation = [torch.relu,torch.tanh]
            irreps_sh = Irreps.spherical_harmonics(lmax, -1)
            ne = n
            no = n

        self.down = Down(n_downsample,activation,irreps_sh,ne,no,batch_norm,input_irreps,kernel_size,down_op,scale,stride,dropout_prob,cutoff)
        ne *= 2**(n_downsample-1)
        no *= 2**(n_downsample-1)
        self.up = Up(n_downsample,activation,irreps_sh,ne,no,batch_norm,self.down.down_irreps_out,kernel_size,up_op,scale,stride,dropout_prob,scalar_upsampling,cutoff)
        self.out = Linear(self.up.up_blocks[-1].irreps_out, output_irreps)

        if is_bias:
            #self.bias = nn.parameter.Parameter(torch.Tensor(n_classes_scalar))
            self.bias = nn.parameter.Parameter(torch.zeros(n_classes_scalar))
        else:
            self.register_parameter('bias', None)



    def forward(self, x):

        def resize(s,n_downsample,odd):

            f = 2**n_downsample
            if odd:
                t = (s - 1) % f
            else:
                t = s % f

            if t != 0:
                s = s + f - t
            return s

        pad = [resize(s,self.n_downsample,self.odd_resize) - s for s in x.shape[-3:]]
        x = torch.nn.functional.pad(x, (pad[-1], 0, pad[-2], 0, pad[-3], 0))


        down_ftrs = self.down(x)
        x = self.up(down_ftrs[-1], down_ftrs)
        x = self.out(x.transpose(1, 4)).transpose(1, 4)

        if self.bias is not None:
            bias = self.bias.reshape(-1, 1, 1, 1)
            x = torch.cat([x[:, :self.n_classes_scalar,...] + bias, x[:, self.n_classes_scalar:,...]], dim=1)

        x = x[..., pad[0]:, pad[1]:, pad[2]:]

        return x
