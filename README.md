###### This repository is a fork of [Neural Geometric Level of Detail](https://github.com/nv-tlabs/nglod) by Takikawa et al.

# Neural LOD for Color and Shape
Master's degree in Computer Science\
Sapienza University of Rome

## Description

The goal of this project is to expand the method for neural shape encoding proposed in [NGLOD](https://github.com/nv-tlabs/nglod) with the encoding of varying surface information, by using the same LOD-based method to generate the implicit representation of a 3D texture.\
The method can be applied to encode any type of surface information defined via 2D textures.
## Results

#### Doom combat scene by [Artec 3D](https://www.artec3d.com/)
<img src="https://user-images.githubusercontent.com/28317156/150802578-f4c8023f-9c46-4f88-9a66-d9f6a356e59a.jpg" width=50% height=50%>

#### Earring
<img src="https://user-images.githubusercontent.com/28317156/150802760-d237d9f0-95f8-4f24-aa5d-d8edd6720e29.jpg" width=50% height=50%>

#### Horn
<img src="https://user-images.githubusercontent.com/28317156/150804352-6b40793a-6aad-454e-83d3-11bb9ac80b15.jpg" width=50% height=50%>

#### Telephone
<img src="https://user-images.githubusercontent.com/28317156/150804428-37e521b0-8183-4aaf-a90e-8bd58b25cbfd.jpg" width=50% height=50%>

#### Veronica
<img src="https://user-images.githubusercontent.com/28317156/150804414-63de4646-7c73-4c4f-8e81-3b9fa617d959.jpg" width=50% height=50%>

#### Violin
<img src="https://user-images.githubusercontent.com/28317156/150804343-c76c7062-921e-4b19-a9b2-26d092bcc675.jpg" width=50% height=50%>

#### Vessel
<img src="https://user-images.githubusercontent.com/28317156/150804395-e538620e-9000-4684-99e6-2e5d57819288.jpg" width=50% height=50%>

#### Pistol
<img src="https://user-images.githubusercontent.com/28317156/150804377-c9063a99-3ccb-4cff-9420-03daf9be2fdf.jpg" width=50% height=50%>

<!-- # Neural Geometric Level of Detail: Real-time Rendering with Implicit 3D Surfaces

Official code release for NGLOD. For technical details, please refer to:

**Neural Geometric Level of Detail: Real-time Rendering with Implicit 3D Surfaces**  
[Towaki Takikawa*](https://tovacinni.github.io), [Joey Litalien*](https://joeylitalien.github.io), [Kangxue Xin](https://kangxue.org/), [Karsten Kreis](https://scholar.google.de/citations?user=rFd-DiAAAAAJ), [Charles Loop](https://research.nvidia.com/person/charles-loop), [Derek Nowrouzezahrai](http://www.cim.mcgill.ca/~derek/), [Alec Jacobson](https://www.cs.toronto.edu/~jacobson/), [Morgan McGuire](https://casual-effects.com/), and [Sanja Fidler](https://www.cs.toronto.edu/~fidler/)\
In Computer Vision and Pattern Recognition (CVPR), 2021 (Oral)\
**[[Paper](https://arxiv.org/abs/2101.10994)] [[Bibtex](https://nv-tlabs.github.io/nglod/assets/nglod.bib)] [[Project Page](https://nv-tlabs.github.io/nglod/)]**

![](imgs/imgs_teaser.jpg)

If you find this code useful, please consider citing:

```
@article{takikawa2021nglod,
    title = {Neural Geometric Level of Detail: Real-time Rendering with Implicit {3D} Shapes}, 
    author = {Towaki Takikawa and
              Joey Litalien and 
              Kangxue Yin and 
              Karsten Kreis and 
              Charles Loop and 
              Derek Nowrouzezahrai and 
              Alec Jacobson and 
              Morgan McGuire and 
              Sanja Fidler},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2021},
}
```

## Directory Structure

`sol-renderer` contains our real-time rnedering code.

`sdf-net` contains our training code.

Within `sdf-net`:

`sdf-net/lib` contains all of our core codebase.

`sdf-net/app` contains standalone applications that users can run.

## Getting started

### Python dependencies
The easiest way to get started is to create a virtual Python 3.8 environment:
```
conda create -n nglod python=3.8
conda activate nglod
pip install --upgrade pip
pip install -r ./infra/requirements.txt
```

The code also relies on [OpenEXR](https://www.openexr.com/), which requires a system library:

```
sudo apt install libopenexr-dev 
pip install pyexr
```

To see the full list of dependencies, see the [requirements](infra/requirements.txt).

### Building CUDA extensions
To build the corresponding CUDA kernels, run:
```
cd sdf-net/lib/extensions
chmod +x build_ext.sh && ./build_ext.sh
```

The above instructions were tested on Ubuntu 18.04/20.04 with CUDA 10.2/11.1.

## Training & Rendering

**Note.** All following commands should be ran within the `sdf-net` directory.

### Download sample data

To download a cool armadillo:

```
wget https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/armadillo.obj -P data/
```

To download a cool matcap file:

```
wget https://raw.githubusercontent.com/nidorx/matcaps/master/1024/6E8C48_B8CDA7_344018_A8BC94.png -O data/matcap/green.png
```

### Training from scratch

```
python app/main.py \
    --net OctreeSDF \
    --num-lods 5 \
    --dataset-path data/armadillo.obj \
    --epoch 250 \
    --exp-name armadillo
```

This will populate `_results` with TensorBoard logs.

### Rendering the trained model

If you set custom network parameters in training, you need to also reflect them for the renderer.

For example, if you set `--feature-dim 16` above, you need to set it here too.

```
python app/sdf_renderer.py \
    --net OctreeSDF \
    --num-lods 5 \
    --pretrained _results/models/armadillo.pth \
    --render-res 1280 720 \
    --shading-mode matcap \
    --lod 4
```

By default, this will populate `_results` with the rendered image.

If you want to export a `.npz` model which can be loaded into the C++ real-time renderer, add the argument 
`--export path/file.npz`. Note that the renderer only supports the base Neural LOD configuration
(the default parameters with `OctreeSDF`).

## Core Library Development Guide

To add new functionality, you will likely want to make edits to the files in `lib`. 

We try our best to keep our code modular, such that key components such as `trainer.py` and `renderer.py` 
need not be modified very frequently to add new functionalities.

To add a new network architecture for an example, you can simply add a new Python file in `lib/models` that
inherits from a base class of choice. You will probably only need to implement the `sdf` method which 
implements the forward pass, but you have the option to override other methods as needed if more custom
operations are needed. 

By default, the loss function used are defined in a CLI argument, which the code will automatically parse
and iterate through each loss function. The network architecture class is similarly defined in the CLI 
argument; simply use the exact class name, and don't forget to add a line in `__init__.py` to resolve the 
namespace.

## App Development Guide

To make apps that use the core library, add the `sdf-net` directory into the Python `sys.path`, so 
the modules can be loaded correctly. Then, you will likely want to inherit the same CLI parser defined
in `lib/options.py` to save time. You can then add a new argument group `app` to the parser to add custom
CLI arguments to be used in conjunction with the defaults. See `app/sdf_renderer.py` for an example.

Examples of things that are considered `apps` include, but are not limited to:

- visualizers
- training code
- downstream applications

## Third-Party Libraries

This code includes code derived from 3 third-party libraries, all distributed under the MIT License:

https://github.com/zekunhao1995/DualSDF

https://github.com/rogersce/cnpy

https://github.com/krrish94/nerf-pytorch

## Acknowledgements

We would like to thank Jean-Francois Lafleche, Peter Shirley, Kevin Xie, Jonathan Granskog, 
Alex Evans, and Alex Bie at NVIDIA for interesting discussions throughout the project. 
We also thank Peter Shirley, Alexander Majercik, Jacob Munkberg, David Luebke, Jonah Philion and 
Jun Gao for their help with paper editing.

We also thank Clement Fuji Tsang for his help with the code release.

The structure of this repo was inspired by PIFu: https://github.com/shunsukesaito/PIFu

 -->
