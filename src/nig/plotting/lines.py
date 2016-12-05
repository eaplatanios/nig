# Copyright 2016, The NIG Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from __future__ import absolute_import, division, print_function

import numpy as np

from matplotlib import pyplot as plt

__author__ = 'eaplatanios'

__all__ = ['plot_lines']


def plot_lines(lines, names=None, style_sheet='ggplot', colormap='viridis',
               xscale='linear', xlabel=None, yscale='linear', ylabel=None,
               title=None, alpha=1.0, linestyle='-', linewidth=2.0, marker=None,
               markersize=0.0, include_legend=True, legend_location='best',
               show_plot=False, save_filename=None, dpi=None, facecolor=None,
               edgecolor=None, orientation='portrait', papertype=None,
               format=None, transparent=False, bbox_inches=None, pad_inches=0.1,
               frameon=None):
    if names is None and isinstance(lines, dict):
        names = list(lines.keys())
        lines = list(lines.values())
    else:
        if set(names) != set(lines.keys()):
            raise ValueError('The provided names are inconsistent with the '
                             'provided lines.')
        lines = [lines[name] for name in names]
    if save_filename is None and not show_plot:
        show_plot = True
    plt.style.use(style_sheet)
    colormap = plt.get_cmap(colormap)
    colors = colormap(np.linspace(0, 1, len(lines)))
    fig = plt.figure()
    for name, line, color in zip(names, lines, colors):
        x, y = zip(*line)
        plt.plot(
            x, y, label=name, color=color, alpha=alpha, linestyle=linestyle,
            linewidth=linewidth, marker=marker, markersize=markersize,
            markeredgewidth=0.0)
    plt.xscale(xscale)
    plt.yscale(yscale)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if include_legend:
        plt.legend(loc=legend_location)
    if show_plot:
        plt.show()
    if save_filename is not None:
        if dpi is None:
            dpi = fig.dpi
        fig.savefig(
            save_filename, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor,
            orientation=orientation, papertype=papertype, format=format,
            transparent=transparent, bbox_inches=bbox_inches,
            pad_inches=pad_inches, frameon=frameon)
