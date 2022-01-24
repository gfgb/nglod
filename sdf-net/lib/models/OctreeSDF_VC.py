# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import math 

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .BaseLOD import BaseLOD
from .BasicDecoder import BasicDecoder
from .utils import init_decoder
from .losses import *
from ..utils import PerfTimer

class MyActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class FeatureVolume(nn.Module):
    def __init__(self, fdim, fsize):
        super().__init__()
        self.fsize = fsize
        self.fdim = fdim
        self.fm = nn.Parameter(torch.randn(1, fdim, fsize+1, fsize+1, fsize+1) * 0.01)
        self.sparse = None

    def forward(self, x):
        N = x.shape[0]
        if x.shape[1] == 3:
            sample_coords = x.reshape(1, N, 1, 1, 3) # [N, 1, 1, 3]    
            sample = F.grid_sample(self.fm, sample_coords, 
                                   align_corners=True, padding_mode='border')[0,:,:,0,0].transpose(0,1)
        else:
            print(x, x.shape)
            print("???")
            sample_coords = x.reshape(1, N, x.shape[1], 1, 3) # [N, 1, 1, 3]    
            sample = F.grid_sample(self.fm, sample_coords, 
                                   align_corners=True, padding_mode='border')[0,:,:,:,0].permute([1,2,0])
        
        return sample


# class NN_vcol(nn.Module):
#     def __init__(self, sdf_input_dim, hidden_dim_sdf, hidden_dim_col):
#         super().__init__()
#         self.sdf_layer = nn.Linear(sdf_input_dim, hidden_dim_sdf, bias=True)
#         self.col_layer = nn.Linear(sdf_input_dim, hidden_dim_col, bias=True)
#         self.sdf_relu = nn.ReLU()
#         self.col_relu = nn.ReLU()
#         self.sdf_output = nn.Linear(hidden_dim_sdf, 1, bias=True)
#         self.col_output = nn.Linear(hidden_dim_col, 3, bias=True)
    
#     def forward_sdf(self, x):
#         d = self.sdf_layer(x)
#         d = self.sdf_relu(d)
#         d = self.sdf_output(d)
#         return d
    
#     def forward_col(self, x):
#         col = self.col_layer(x)
#         col = self.col_relu(col)
#         col = self.col_output(col)
#         return col
    
#     def forward(self, x):
#         d = self.forward_sdf(x)
#         col = self.forward_col(x)
#         return torch.cat((d, col), dim=1)

class NN_vcol(nn.Module):
    def __init__(self, sdf_input_dim, hidden_dim_sdf, hidden_dim_col):
        super().__init__()
        self.nn_sdf = nn.Sequential(
            nn.Linear(sdf_input_dim, hidden_dim_sdf, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim_sdf, 1, bias=True),
        )
        self.nn_col = nn.Sequential(
            nn.Linear(sdf_input_dim, hidden_dim_col, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim_col, 3, bias=True),
            nn.Sigmoid()
        )
        self.nns = (self.nn_sdf, self.nn_col)
    
    # def forward_sdf(self, x):
    #     return self.nn_sdf(x)
    
    # def forward_col(self, x):
    #     return self.nn_col(x)
    
    # def forward(self, x):
    #     d = self.forward_sdf(x)
    #     col = self.forward_col(x.clone())
    #     return d, col

        # return torch.cat((d, col), dim=1)

class FeatureVolumes(nn.Module):
    def __init__(self, fdim, fsize, i):
        super().__init__()
        self.fsize = fsize
        self.fdim = fdim
        self.fv_d = FeatureVolume(fdim, fsize * (2**i))
        self.fv_col = FeatureVolume(fdim, fsize * (2**i))
        self.fvs = (self.fv_d, self.fv_col)




class OctreeSDF_VC(BaseLOD):
    def __init__(self, args, init=None):
        super().__init__(args)

        self.fdim = self.args.feature_dim
        self.fsize = self.args.feature_size
        self.hidden_dim = self.args.hidden_dim
        self.pos_invariant = self.args.pos_invariant

        self.features = nn.ModuleList([])
        for i in range(self.args.num_lods):
            # self.features.append(FeatureVolume(self.fdim, self.fsize * (2**i)))
            self.features.append(
                # (FeatureVolume(self.fdim, self.fsize * (2**i)), 
                # FeatureVolume(self.fdim, self.fsize * (2**i))
                # )
                FeatureVolumes(self.fdim, self.fsize, i)
            )
    
        self.interpolate = self.args.interpolate

        self.louts = nn.ModuleList([])

        self.sdf_input_dim = self.fdim
        if not self.pos_invariant:
            self.sdf_input_dim += self.input_dim

        self.num_decoder = 1 if args.joint_decoder else self.args.num_lods 

        for i in range(self.num_decoder):
            self.louts.append(
                # (nn.Sequential(
                #     nn.Linear(self.sdf_input_dim, self.hidden_dim, bias=True),
                #     nn.ReLU(),
                #     nn.Linear(self.hidden_dim, 1, bias=True),
                # ),
                # nn.Sequential(
                #     nn.Linear(self.sdf_input_dim, self.hidden_dim, bias=True),
                #     nn.ReLU(),
                #     nn.Linear(self.hidden_dim, 3, bias=True),
                # ))
                NN_vcol(self.sdf_input_dim, self.hidden_dim, self.hidden_dim)
            )
        
    def encode(self, x):
        # Disable encoding
        return x

    def asdf(self, x, fvol_idx : int, lod=None):

        l = []
        samples = []

        for i in range(self.num_lods):

            sample = self.features[i].fvs[fvol_idx](x)
            samples.append(sample)

            # Sum queried features
            if i > 0:
                samples[i] += samples[i-1]
            
            # Concatenate xyz
            ex_sample = samples[i]
            if not self.pos_invariant:
                ex_sample = torch.cat([x, ex_sample], dim=-1)

            if self.num_decoder == 1:
                prev_decoder = self.louts[0].nns[fvol_idx]
                curr_decoder = self.louts[0].nns[fvol_idx]
            else:
                prev_decoder = self.louts[i-1].nns[fvol_idx]
                curr_decoder = self.louts[i].nns[fvol_idx]
            
            res = curr_decoder(ex_sample)

            # Interpolation mode
            if self.interpolate is not None and lod is not None:

                print("???")
            
            # Get distance
            else: 
                res = curr_decoder(ex_sample)

                # Return distance if in prediction mode
                if lod is not None and lod == i:
                    return res

                l.append(res)
        
        return l[-1]

    def sdf(self, x, lod=None, return_lst=False):

        if lod is None:
            lod = self.lod

        # Query
        d = self.asdf(x, fvol_idx=0, lod=lod)
        col = self.asdf(x, fvol_idx=1, lod=lod)

        if self.training:
            self.loss_preds = (d, col)

        return d, col
    

    # def sdf(self, x, lod=None, return_lst=False):
    #     if lod is None:
    #         lod = self.lod

    #     # Query
    #     l = []
    #     samples = []

    #     for i in range(self.num_lods):
            
    #         # Query features
    #         sample = self.features[i](x)
    #         samples.append(sample)

    #         # Sum queried features
    #         if i > 0:
    #             samples[i] += samples[i-1]
            
    #         # Concatenate xyz
    #         ex_sample = samples[i]
    #         if not self.pos_invariant:
    #             ex_sample = torch.cat([x, ex_sample], dim=-1)

    #         if self.num_decoder == 1:
    #             prev_decoder = self.louts[0]
    #             curr_decoder = self.louts[0]
    #         else:
    #             prev_decoder = self.louts[i-1]
    #             curr_decoder = self.louts[i]
            
    #         d, col = curr_decoder(ex_sample)

    #         # Interpolation mode
    #         if self.interpolate is not None and lod is not None:

    #             print("???")
                
    #             if i == len(self.louts) - 1:
    #                 return d

    #             if lod+1 == i:
    #                 _ex_sample = samples[i-1]
    #                 if not self.pos_invariant:
    #                     _ex_sample = torch.cat([x, _ex_sample], dim=-1)
    #                 _d = prev_decoder(_ex_sample)

    #                 return (1.0 - self.interpolate) * _l + self.interpolate * d
            
    #         # Get distance
    #         else: 
    #             d, col = curr_decoder(ex_sample)

    #             # Return distance if in prediction mode
    #             if lod is not None and lod == i:
    #                 return d, col

    #             l.append((d, col))
    #     if self.training:
    #         self.loss_preds = (d, col)

    #     if return_lst:
    #         return l
    #     else:
    #         return l[-1]
