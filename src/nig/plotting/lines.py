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


def plot_lines(lines, names=None, style='ggplot', colormap='viridis',
               xscale='linear', xlabel=None, yscale='linear', ylabel=None,
               title=None, font_size=16, alpha=1.0, linestyle='-',
               linewidth=2.0, marker=None, markersize=0.0, include_legend=True,
               legend_location='best', show_plot=False, save_filename=None,
               dpi=None, transparent=True, bbox_inches='tight', pad_inches=0.1,
               figure=None, axes=None):
    if names is None and isinstance(lines, dict):
        names = list(lines.keys())
        lines = list(lines.values())
    elif isinstance(lines, dict):
        if set(names) != set(lines.keys()):
            raise ValueError('The provided names are inconsistent with the '
                             'provided lines.')
        lines = [lines[name] for name in names]
    # if save_filename is None and not show_plot:
    #     show_plot = True
    with plt.style.context(style):
        colormap = plt.get_cmap(colormap)
        colors = colormap(np.linspace(0, 1, len(lines)))
        if figure is None and axes is None:
            figure, axes = plt.subplots()
        elif figure is None:
            figure = plt.figure()
        for name, line, color in zip(names, lines, colors):
            x, y = zip(*line)
            axes.plot(
                x, y, label=name, color=color, alpha=alpha, linestyle=linestyle,
                linewidth=linewidth, marker=marker, markersize=markersize,
                markeredgewidth=0.0)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        if xlabel is not None:
            axes.set_xlabel(xlabel, fontdict={'fontsize': font_size})
        if ylabel is not None:
            axes.set_ylabel(ylabel, fontdict={'fontsize': font_size})
        if title is not None:
            axes.set_title(title, fontdict={'fontsize': font_size + 2}, y=1.03)
        if include_legend:
            axes.legend(loc=legend_location, fontsize=font_size - 2)
        if show_plot:
            plt.show()
        if save_filename is not None:
            figure.savefig(
                save_filename, dpi=dpi, transparent=transparent,
                bbox_inches=bbox_inches, pad_inches=pad_inches)
