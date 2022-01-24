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

import os

import torch
from torch.utils.data import Dataset

from PIL import Image
import torchvision

import numpy as np
import mesh2sdf
from collections import Counter

from lib.torchgp import load_obj, point_sample, sample_surface, compute_sdf, normalize
from lib.PsDebugger import PsDebugger

from lib.torchgp.compute_sdf_tex import compute_sdf_tex

from lib.utils import PerfTimer, setparam


class MeshDataset_TEX(Dataset):
    """Base class for single mesh datasets."""

    def __init__(
        self,
        args=None,
        dataset_path=None,
        raw_obj_path=None,
        sample_mode=None,
        get_normals=None,
        seed=None,
        num_samples=None,
        trim=None,
        sample_tex=None,
    ):
        self.args = args
        self.dataset_path = setparam(args, dataset_path, "dataset_path")
        self.vcol_path = ".".join(self.dataset_path.split(".")[:-1]) + ".vcol"

        self.raw_obj_path = setparam(args, raw_obj_path, "raw_obj_path")
        self.sample_mode = setparam(args, sample_mode, "sample_mode")
        self.get_normals = setparam(args, get_normals, "get_normals")
        self.num_samples = setparam(args, num_samples, "num_samples")
        self.trim = setparam(args, trim, "trim")
        self.sample_tex = setparam(args, sample_tex, "sample_tex")

        self.samples = self.load_samples() if args.use_precomputed_samples else None

        # Possibly remove... or fix trim obj
        # if self.raw_obj_path is not None and not os.path.exists(self.dataset_path):
        #    _, _, self.mesh = trim_obj_to_file(self.raw_obj_path, self.dataset_path)
        # elif not os.path.exists(self.dataset_path):
        #    assert False and "Data does not exist and raw obj file not specified"
        # else:

        # if self.sample_tex:
        #     out = load_obj(self.dataset_path, load_materials=True)
        #     self.V, self.F, self.texv, self.texf, self.mats = out
        # else:
        #     self.V, self.F = load_obj(self.dataset_path)

        out = load_obj(self.dataset_path, load_materials=True)
        self.V, self.F, self.texv, self.texf, self.mats = out
        self.texf = self.texf[:, :-1]

        if self.texv.shape[0] == 0:
            self.texv = torch.zeros((self.V.shape[0], 2), dtype=torch.float32)
            self.texf = self.F

        self.texture = Image.open(
            "/".join(self.dataset_path.split(".")[:-1]) + ".png"
        ).convert("RGB")
        self.texture = torchvision.transforms.ToTensor()(self.texture)
        self.texture = self.texture.permute(1, 2, 0)

        self.V, self.F = normalize(self.V, self.F)

        self.mesh = self.V[self.F]

        self.resample()
        
    def sample_sdf(self, mode: list = None):

        nrm = None
        if self.get_normals:
            pts, nrm = sample_surface(self.V, self.F, self.num_samples * 5)
            nrm = nrm.cpu()
        else:
            sample_mode = self.sample_mode if mode is None else mode
            pts = point_sample(self.V, self.F, sample_mode, self.num_samples)

        d, col = compute_sdf_tex(
            self.V.cuda(),
            self.F.cuda(),
            pts.cuda(),
            self.texv.cuda(),
            self.texf.cuda(),
            self.texture.cuda(),
        )

        return pts, d, col

    def sample_and_save(self, modes, n_samples, filename):
        
        # n_samples * 100000

        res = {}

        for mode, ns in zip(modes, n_samples):

            for i in range(ns):
                
                print(f"Computing {mode} {i+1}/{ns}")

                pts, d, col = self.sample_sdf([mode])
                pts, d, col = pts.cpu(), d.cpu(), col.cpu()

                if mode not in res:
                    res[mode] = { "pts": pts, "d": d, "col": col }
                else:
                    res[mode]["pts"] = torch.cat((res[mode]["pts"], pts), dim=0)
                    res[mode]["d"]   = torch.cat((res[mode]["d"], d), dim=0)
                    res[mode]["col"] = torch.cat((res[mode]["col"], col), dim=0)
        
        torch.save(res, filename)
        return res

    def load_samples(self):
        filename = f"data/samples/{self.args.exp_name.split('/')[-1]}.pt"
        try:
            data = torch.load(filename, map_location="cpu")
        except:
            print(f"Samples file {filename} not found")
            exit(1)
        return data
    
    def draw_samples(self):

        n_samples_to_draw = {k:v * 100000 for k, v in Counter(self.sample_mode).items()}

        pts, d, col = torch.Tensor(), torch.Tensor(), torch.Tensor()
        
        for mode in n_samples_to_draw:
            
            idx = torch.from_numpy(np.random.choice(self.samples[mode]["pts"].shape[0], (n_samples_to_draw[mode],), replace=False)).long()

            pts = torch.cat((pts, self.samples[mode]["pts"][idx]), dim=0)
            d   = torch.cat((d, self.samples[mode]["d"][idx]), dim=0)
            col = torch.cat((col, self.samples[mode]["col"][idx]), dim=0)

        shuffle = torch.randperm(sum([ n_samples_to_draw[x] for x in n_samples_to_draw ]), dtype=torch.int64)

        return pts[shuffle], d[shuffle], col[shuffle]
    
    def plot_points(self):

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False, elev=30)
        fig.add_axes(ax)
        ax.set_box_aspect((1, 1, 1))
        ax.scatter(self.pts[:, 0], self.pts[:, 2], self.pts[:, 1], s=0.1, c=self.col.cpu())
        
        # ax.scatter([p[0], cp[0]], [p[1], cp[1]], [p[2], cp[2]], s=100, c="red")
        # ax.plot([p[0], cp[0]], [p[1], cp[1]], [p[2], cp[2]])
        # ax.plot([p[0], 0], [p[1], 0])
        
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)

        plt.show()
        exit(1)

    def resample(self):
        """Resample SDF samples."""

        # self.sample_and_save(["rand", "near", "trace"], [15, 30, 30], f"data/samples/{self.args.exp_name.split('/')[-1]}.pt")

        if self.args.use_precomputed_samples:
            self.pts, self.d, self.col = self.draw_samples()
        else:
            self.pts, self.d, self.col = self.sample_sdf()

        # self.plot_points()

        self.d = self.d[..., None].cpu()
        self.d = self.d.cpu()
        self.col = self.col.cpu()
        self.pts = self.pts.cpu()
        
    
    def __getitem__(self, idx: int):
        """Retrieve point sample."""
        if self.get_normals:
            return self.pts[idx], self.d[idx], self.nrm[idx]
        elif self.sample_tex:
            return self.pts[idx], self.d[idx], self.rgb[idx]
        else:
            return self.pts[idx], self.d[idx], self.col[idx]

    def __len__(self):
        """Return length of dataset (number of _samples_)."""
        return self.pts.size()[0]

    def num_shapes(self):
        """Return length of dataset (number of _mesh models_)."""
        return 1
