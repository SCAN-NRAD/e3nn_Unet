# About e3nn_Unet

Segmentation networks based on e3nn O3 and SO3 equivariant convolutions.
The networks allow for scalar, vector or tensor inputs, user-defined number and 
type of downsampling operations, type of equivariance, and number of hidden
irreps.

## Installation

```bash
git clone https://github.com/SCAN-NRAD/e3nn_Unet.git
cd e3nn_Unet
pip install -e .
```

## Sample UNet

The following commands create a U-net for a segmentation task with two 
input scalar channels, 2 downsampling steps, and SO3 equivariance
and which outputs 5 scalar classes. In this example the kernel diameter 
is set to five units and the voxel dimensions are 1x1x1 units.

```bash
from e3nn_Unet.network.equivariant_unet_physical_units import UNet
net = UNet('2x0e','5x0e',5,5,(1,1,1),n_downsample=2,equivariance='SO3')
```
