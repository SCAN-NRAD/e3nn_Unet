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

The following commands create a U-net for a segmentation task with three 
scalar classes, 0 vector classes, 2 downsampling steps, and S03 equivariance.

```bash
from e3nn_Unet.network.equivariant_unet import UNet
net = equivariant_unet.UNet(3,0,n_downsample=2,equivariance='SO3') 
```
