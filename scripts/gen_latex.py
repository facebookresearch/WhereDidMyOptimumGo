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

def better_table(table, caption, name, transpose=False):
    
    start = r"""
\begin{table}[H]
\centering
\small{
"""

    middle = """
}}
\caption{{{}}}\label{{table:{}}}
""".format(caption, name)
    
    end = r"\end{table}"
    numerical_format = r'\num{{{}}}'.format
    table = table.transpose()
    table = table.reindex_axis([table.columns[0]] + sorted(table.columns[1:]), axis=1)
    table_latex = table.to_latex(
        escape=False,
        index=False,
        column_format="|" + "|".join(["c"] *table.shape[1])+ "|")
    
    for str_to_replace in ['midrule', 'toprule', 'bottomrule']:
        table_latex = table_latex.replace(str_to_replace, 'hline')
    table_latex= table_latex.replace(".00", "")
    table_latex= table_latex.replace(".0 ", "")
    
    return start + table_latex +  middle + end

def gen_caption(dirname, tag):
    if "asymptotic" in dirname:
        asymptotic = True
        as_name = "asymptotic"
    else:
        as_name = "average"
        asymptotic = False
   
    algo = "a2c" if "a2c" in dirname else "ppo"

    base_caption = "{} {} performance across various momentum values with SGD and Nesterov Momentum.".format(algo.upper(), as_name)
    std_or_average = "Average" if tag is "avg" else "Standard Deviation"
    if asymptotic:
        base_caption += " {} returns over the final 50 training episodes across 10 random seeds.".format(std_or_average)
    else:
        base_caption += " {} returns over all training episodes over 10 random seeds.".format(std_or_average)
    return base_caption, algo + "-" + as_name + "-" + tag 

def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('indir', help="Input file", type=str)
    parser.add_argument("--transpose", action="store_true")

    args = parser.parse_args(arguments)

    current_data = None
    dfs = {}

    for filename in sorted(os.listdir(args.indir)):
        if filename.endswith(".csv"):
            if "avg" in filename and "avg" in dfs:
                current_data = dfs["avg"]
            elif "std" in filename and "std" in dfs:
                current_data = dfs["std"]
            else:
                current_data = None
         
            data = pandas.read_csv(args.indir + "/" + filename, delimiter="&")
            copy = data.columns.values
            env = filename.split("-v2")[0]
            copy[1] = env
            copy[0] = "Momentum"
            data.columns = copy
            if current_data is None:
                single_data = data.transpose()
                current_data = single_data
            else:
                current_data = current_data.append(data[env].transpose())
            if "avg" in filename:
                dfs["avg"] = current_data
            if "std" in filename:
                dfs["std"] = current_data
        else:
            continue
    for key, val in dfs.items():
        with open(os.path.join(args.indir, key + ".tex"), 'w') as f:
            caption, tag = gen_caption(args.indir, key)
            f.write(better_table(val, caption, tag, args.transpose))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
