import math
from functools import partial
import torch
from torch import nn
from e3nn.nn import BatchNorm, Gate, Dropout
from e3nn.o3 import Irreps, Linear
import numpy as np
from nnunet.network_architecture.neural_network import SegmentationNetwork
from .voxel_convolution import Convolution
from .pooling import DynamicPool3d


class ConvolutionBlock(nn.Module):
    def __init__(self, input, irreps_hidden, activation, irreps_sh, normalization,diameter,num_radial_basis,steps,dropout_prob,cutoff):
        super().__init__()

        if normalization == 'None':
            BN = Identity
        elif normalization == 'batch':
            BN = BatchNorm
        elif normalization == 'instance':
            BN = partial(BatchNorm,instance=True)


        irreps_scalars = Irreps( [ (mul, ir) for mul, ir in irreps_hidden if ir.l == 0 ] )
        irreps_gated   = Irreps( [ (mul, ir) for mul, ir in irreps_hidden if ir.l > 0  ] )
        irreps_gates = Irreps(f"{irreps_gated.num_irreps}x0e")

        #fe = sum(mul for mul,ir in irreps_gated if ir.p == 1)
        #fo = sum(mul for mul,ir in irreps_gated if ir.p == -1)
        #irreps_gates = Irreps(f"{fe}x0e+{fo}x0o").simplify()
        if irreps_gates.dim == 0:
            irreps_gates = irreps_gates.simplify()
            activation_gate = []
        else:
            activation_gate = [torch.sigmoid]
            #activation_gate = [torch.sigmoid, torch.tanh][:len(activation)]

        self.gate1 = Gate(irreps_scalars, activation, irreps_gates, activation_gate, irreps_gated)
        self.conv1 = Convolution(input, self.gate1.irreps_in, irreps_sh, diameter,num_radial_basis,steps,cutoff=cutoff)
        self.batchnorm1 = BN(self.gate1.irreps_in)
        self.dropout1 = Dropout(self.gate1.irreps_out, dropout_prob)

        self.gate2 = Gate(irreps_scalars, activation, irreps_gates, activation_gate, irreps_gated)
        self.conv2 = Convolution(self.gate1.irreps_out, self.gate2.irreps_in, irreps_sh, diameter,num_radial_basis,steps,cutoff=cutoff)
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

    def __init__(self, n_downsample,activation,irreps_sh,ne,no,BN,input,diameters,num_radial_basis,steps,down_op,scale,stride,dropout_prob,cutoff):
        super().__init__()

        blocks = []
        self.down_irreps_out = []

        for n in range(n_downsample+1):
            irreps_hidden = Irreps(f"{4*ne}x0e + {4*no}x0o + {2*ne}x1e +  {2*no}x1o + {ne}x2e + {no}x2o").simplify()
            block = ConvolutionBlock(input,irreps_hidden,activation,irreps_sh,BN, diameters[n],num_radial_basis,steps[n],dropout_prob,cutoff)
            blocks.append(block)
            self.down_irreps_out.append(block.irreps_out)
            input = block.irreps_out
            ne *= 2
            no *= 2

        self.down_blocks = nn.ModuleList(blocks)

        pooling = []
        for n in range(n_downsample):
            pooling.append(DynamicPool3d(scale[n],steps[n],down_op,self.down_irreps_out[n]))

        self.down_pool = nn.ModuleList(pooling)

    def forward(self, x):
        ftrs = []
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            ftrs.append(x)
            if i < len(self.down_blocks)-1:
                x = self.down_pool[i](x)
        return ftrs

class Up(nn.Module):
    def __init__(self, n_blocks_up,activation,irreps_sh,ne,no,BN,downblock_irreps,diameters,num_radial_basis,steps,up_op,scale,stride,dropout_prob,scalar_upsampling,cutoff):
        super().__init__()

        self.n_blocks_up = n_blocks_up
        #if up_op == 'lowpass':
        #    self.upsamp    = LowPassFilter(scale,stride=stride,transposed=True,steps=steps)
        #else:
        #    self.upsamp = nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=True)

        input = downblock_irreps[-1]
        blocks = []
        upsample_op = []

        for n in range(n_blocks_up):
            if scalar_upsampling:
                irreps_hidden = Irreps(f"{8*ne}x0e+{8*no}x0o").simplify()
            else:
                irreps_hidden = Irreps(f"{4*ne}x0e + {4*no}x0o + {2*ne}x1e + {ne}x2e + {2*no}x1o + {no}x2o").simplify()

            block = ConvolutionBlock(input+downblock_irreps[::-1][n+1],irreps_hidden,activation,irreps_sh,BN,diameters[n],num_radial_basis,steps[n],dropout_prob,cutoff)
            blocks.append(block)
            input = block.irreps_out
            ne //= 2
            no //= 2

            upsample_scale_factor = tuple([math.floor(scale[n]/step) if step < scale[n] else 1 for step in steps[n]]) #same as pooling kernel
            upsample_op.append(nn.Upsample(scale_factor=upsample_scale_factor, mode='trilinear', align_corners=True))

        self.up_blocks = nn.ModuleList(blocks)
        self.upsample_ops = nn.ModuleList(upsample_op)


    def forward(self, x, down_features):

        for i in range(self.n_blocks_up):
            x        = self.upsample_ops[i](x)
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
    def __init__(self, input_irreps, output_irreps, diameter, num_radial_basis, steps, batch_norm='instance', n=2, n_downsample = 2, equivariance = 'SO3',
        lmax = 2, down_op = 'maxpool3d', stride = 2, scale =2,,
        is_bias = True,scalar_upsampling=False,dropout_prob=0,cutoff=False):
        """Equivariant UNet with physical units

        Parameters
        ----------
        input_irreps : str
            input representations
            example: "1x0e" when one channel of scalar values
        output_irreps : str
            output representations
            example: "4x0e" when four channels of scalar values
        n_classes_vector : int
            number of vector classes
        diameter : float
            diameter of input convolution kernel in physical units
        num_radial_basis : int
            number of radial basis functions
        steps : float
            physical dimension of a pixel in physical units
        batch_norm : str, optional
            normalization: can be 'batch', 'instance' or 'None'.
            by default 'instance'
        n : int, optional
            multiplication factor of number of irreps
            between successive convolution blocks, by default 2
        n_downsample : int, optional
            number of downsampling operations, by default 2
        equivariance : str, optional
            type of equivariance, can be 'O3' or 'SO3'
            by default 'SO3'
        lmax : int, optional
            maximum spherical harmonics l
            by default 2
        down_op : str, optional
            type of downsampling operation
            can be 'maxpool3d', 'average' or 'lowpass'
            by default 'maxpool3d'
        stride : int, optional
            stride size, by default 2
        scale : int, optional
            size of pooling diameter
            in physical units, by default 2
        is_bias : bool, optional
            defines whether or not to add a bias, by default True
        scalar_upsampling : bool, optional
            flag to use scalar_upsampling, by default False
        dropout_prob : float, optional
            dropout probability between 0 and 1.0, by default 0
        cutoff: bool, optional
            cutoff basis functions at 0 outside of [0,r], by default False

        """
        super().__init__()

        self.n_classes_scalar = Irreps(output_irreps).count('0e')
        self.num_classes = self.n_classes_scalar
        
        self.n_downsample = n_downsample
        self.conv_op = nn.Conv3d #Needed in order to use nnUnet predict_3D

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
        scales = [scale*2**i for i in range(n_downsample)] #TODO change 2 to variable factor
        diameters = [diameter*2**i for i in range(n_downsample+1)] #TODO change 2 to variable factor

        steps_array = [steps]
        for i in range(n_downsample):
            
            output_steps = []
            for step in steps:
                if step < scales[i]:
                    kernel_dim = math.floor(scales[i]/step)
                    output_steps.append(kernel_dim*step)
                else:
                    output_steps.append(step)

            steps_array.append(tuple(output_steps))


        self.down = Down(n_downsample,activation,irreps_sh,ne,no,batch_norm,input_irreps,diameters,num_radial_basis,steps_array,down_op,scales,stride,dropout_prob,cutoff)
        ne *= 2**(n_downsample-1)
        no *= 2**(n_downsample-1)
        self.up = Up(n_downsample,activation,irreps_sh,ne,no,batch_norm,self.down.down_irreps_out,diameters[::-1][1:], num_radial_basis,steps_array[::-1][1:],up_op,scales[::-1],stride,dropout_prob,scalar_upsampling,cutoff)
        self.out = Linear(self.up.up_blocks[-1].irreps_out, output_irreps)

        if is_bias:
            #self.bias = nn.parameter.Parameter(torch.Tensor(n_classes_scalar))
            self.bias = nn.parameter.Parameter(torch.zeros(n_classes_scalar))
        else:
            self.register_parameter('bias', None)


    def forward(self, x):

        def pad_size(image_shape,odd):

            pooling_factor = np.ones(3,dtype='int')
            for pool in self.down.down_pool:
                pooling_factor *= np.array(pool.kernel_size)

            pad = [] 
            for f,s in zip(pooling_factor,image_shape):
                p = 0  #padding for current dimension
                if odd:
                    t = (s - 1) % f
                else:
                    t = s % f

                if t != 0:
                    p = f - t
                pad.append(p)
            
            return pad

        pad = pad_size(x.shape[-3:],self.odd_resize)
        x = torch.nn.functional.pad(x, (pad[-1], 0, pad[-2], 0, pad[-3], 0))

        down_ftrs = self.down(x)
        x = self.up(down_ftrs[-1], down_ftrs)
        x = self.out(x.transpose(1, 4)).transpose(1, 4)

        if self.bias is not None:
            bias = self.bias.reshape(-1, 1, 1, 1)
            x = torch.cat([x[:, :self.n_classes_scalar,...] + bias, x[:, self.n_classes_scalar:,...]], dim=1)
        
        x = x[..., pad[0]:, pad[1]:, pad[2]:]

        return x
