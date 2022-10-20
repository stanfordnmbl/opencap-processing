'''
    ---------------------------------------------------------------------------
    OpenCap processing: utilsPlotting.py
    ---------------------------------------------------------------------------

    Copyright 2022 Stanford University and the Authors
    
    Author(s): Antoine Falisse, Scott Uhlrich
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
'''

import numpy as np
import matplotlib.pyplot as plt

def plot_dataframe(dataframes, x=None, y=[], xlabel=None, ylabel=None, 
                   labels=None, title=None, xrange=None):
    
    # Handle case specific number of subplots.
    if not x and not y:
        nRow = int(np.ceil(np.sqrt(dataframes[0].shape[1]-1)))
        nCol = int(np.ceil(np.sqrt(dataframes[0].shape[1]-1)))
        if not xlabel:
            xlabel = list(dataframes[0].columns)[0]
        x = 'time'
        y = list(dataframes[0].columns)[1:]        
    elif not x and y:
        nRow = int(np.ceil(np.sqrt(len(y))))
        nCol = int(np.ceil(np.sqrt(len(y))))
        if not xlabel:
            xlabel = list(dataframes[0].columns)[0]
        x = 'time'
    else:
        nRow = int(np.ceil(np.sqrt(len(y))))
        nCol = int(np.ceil(np.sqrt(len(y))))
        if not xlabel:
            xlabel = x
        if not ylabel:
            ylabel = y[0]        
    if nRow >= len(y):
        nRow = 1
    nAxs = len(y)
        
    # Labels for legend.
    if not labels:
        labels = ['dataframe_' + str(i) for i in range(len(dataframes))]
    elif len(labels) != len(dataframes):
        print("WARNING: The number of labels ({}) does not match the number of input dataframes ({})".format(len(labels), len(dataframes)))
        labels = ['dataframe_' + str(i) for i in range(dataframes)]
 
    if nCol == 1: # Single plot.
        fig = plt.figure()
        color=iter(plt.cm.rainbow(np.linspace(0,1,len(dataframes)))) 
        for c, dataframe in enumerate(dataframes):
            c_color = next(color)     
            plt.plot(dataframe[x], dataframe[y], c=c_color, label=labels[c])
            if xrange is not None:
                plt.xlim(xrange)
    else: # Multiple subplots.
        fig, axs = plt.subplots(nRow, nCol, sharex=True)     
        for i, ax in enumerate(axs.flat):
            color=iter(plt.cm.rainbow(np.linspace(0,1,len(dataframes)))) 
            if i < nAxs:
                for c, dataframe in enumerate(dataframes):
                    c_color = next(color)                
                    ax.plot(dataframe[x], dataframe[y[i]], c=c_color, label=labels[c])
                    ax.set_title(y[i])
                    if xrange is not None:
                        plt.xlim(xrange)
            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
        
    # Axis labels and legend.
    if nRow > 1 and nCol > 1:
        plt.setp(axs[-1, :], xlabel=xlabel)
        plt.setp(axs[:, 0], ylabel=ylabel)
        axs[0][0].legend(handles, labels)
    elif nRow == 1 and nCol > 1:
        plt.setp(axs[:,], xlabel=xlabel)
        plt.setp(axs[0,], ylabel=ylabel)
        axs[0,].legend(handles, labels)
    else:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(labels)
        
    if nRow == 1 and nCol == 1:
        # Add figure title.
        if title:
            plt.title(title)        
    else:
        # Add figure title.
        if title:
            fig.suptitle(title)
        # Align labels.        
        fig.align_ylabels()
        # Hidde empty subplots.
        nEmptySubplots = (nRow*nCol) - len(y)
        axs_flat = axs.flat
        for ax in (axs_flat[len(axs_flat)-nEmptySubplots:]):
            ax.set_visible(False)
                   
    # Tight layout (should make figure big enough first).
    # fig.tight_layout()
    
    # Show plot (needed if running through terminal).
    plt.show()
    