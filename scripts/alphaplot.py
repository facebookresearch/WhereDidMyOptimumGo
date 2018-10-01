#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""A simple python script template.

"""

from __future__ import print_function
import os
import sys
import argparse
import pandas
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({
    'figure.autolayout': True,
    'axes.labelsize': 'xx-large',
    'axes.titlesize' : 'xx-large',
    'xtick.labelsize': 'x-large',
    'ytick.labelsize': 'x-large',
    'legend.fontsize' : 'small',
    'font.size': 14,
    'font.family': 'Times New Roman',
    'font.serif': 'Times',
    'xtick.major.width' : 3,
    'xtick.minor.width' : 3,
    'axes.labelpad': 20,
    'axes.ymargin' : 0.05,
    'axes.xmargin' : 0.05
})
def categorical_cmap(nc, nsc, cmap="tab20"):
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv,nsc).reshape(nsc,3)
        arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
        arhsv[:,2] = np.linspace(chsv[2],1,nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc,:] = rgb
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap

def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('indir', help="Input file", type=str)
    parser.add_argument('--momentum', action="store_true")

    args = parser.parse_args(arguments)

    for filename in os.listdir(args.indir):
        if filename.endswith(".csv") and "avg" in filename:
            print(args)
            data = pandas.read_csv(args.indir + "/" + filename, delimiter="&")

            new_columns = data.columns.values
            new_columns[0] = 'lr'
            data.columns = new_columns
            nolr = data.drop(['lr'], axis=1)
            if "reacher" in filename.lower():
                ylim = (-500, 0)
            else:
                ylim = (-1500, 4000)

            colors_needed = len(data['lr'].values.reshape(-1))
            shades_per_color = min(int(colors_needed/20), 1)
            if args.momentum:
                axes = nolr.plot(colors=categorical_cmap(max(colors_needed, 20), shades_per_color), x=data['lr'].values.reshape(-1), ylim=ylim, figsize=(16,8), linewidth=4)
                axes.set_ylabel('Returns')
                axes.set_xlabel('Momentum')
            else:
                axes = nolr.plot(colormap='tab20', logx=True, x=data['lr'].values.reshape(-1), ylim=ylim, figsize=(16,8), linewidth=4)
                axes.set_ylabel('Returns')
                axes.set_xlabel('Learning Rate')
            plt.title(filename.split("-v2")[0])
            outfile = args.indir + "/" + filename.split(".")[0] + ".pdf"
            plt.savefig(outfile, bbox_inches='tight')
        else:
            continue



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
