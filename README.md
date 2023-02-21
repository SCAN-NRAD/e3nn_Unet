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

## Segmentation of brain structures

The following animation shows the segmentation of seven brain structures 
from the Mindboggle 101 dataset. Each row contains three views of a single segmentation.
Each model segmented the same case rotated by angles from O to 180 degrees.
The equivariant model (e3nn) is stable as the rotation angle changes. The non-equivariant
model with full rotation data augmentation is also exhibits a stable segmentation but exhibits
problems with the hippocampus and brain stem. Finally the two bottom rows show unstable segmentations
for both a non-equivariant model trained with moderate rotational equivariance and a model trained with
no rotational equivariance.

![](https://github.com/SCAN-NRAD/e3nn_Unet/blob/main/comparison.gif ) 
