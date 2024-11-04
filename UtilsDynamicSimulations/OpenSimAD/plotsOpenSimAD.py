'''
    ---------------------------------------------------------------------------
    OpenCap processing: plotsOpenSimAD.py
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

# TODO: This script is poorly documented and generates rough plots for 
# debugging.

import numpy as np

from utilsOpenSimAD import plotVSBounds
from utilsOpenSimAD import plotVSvaryingBounds

def plotGuessVSBounds(lw, uw, w0, nJoints, N, d, guessQsEnd, 
                      guessQdsEnd, withArms=True, 
                      withLumbarCoordinateActuators=True,
                      torque_driven_model=False):
    
    # States.
    if torque_driven_model:
        # Coordinate activation at mesh points.
        lwp = lw['CoordA'].to_numpy().T
        uwp = uw['CoordA'].to_numpy().T
        y = w0['CoordA'].to_numpy().T
        title='Coordinate activation at mesh points' 
        plotVSBounds(y,lwp,uwp,title) 
        # Coordinate activation at collocation points.
        lwp = lw['CoordA'].to_numpy().T
        uwp = uw['CoordA'].to_numpy().T
        y = w0['CoordAj'].to_numpy().T
        title='Coordinate activation at collocation points' 
        plotVSBounds(y,lwp,uwp,title)
    else:
        # Muscle activation at mesh points.    
        lwp = lw['A'].to_numpy().T
        uwp = uw['A'].to_numpy().T
        y = w0['A'].to_numpy().T
        title='Muscle activation at mesh points'            
        plotVSBounds(y,lwp,uwp,title)  
        # Muscle activation at collocation points.
        lwp = lw['A'].to_numpy().T
        uwp = uw['A'].to_numpy().T
        y = w0['Aj'].to_numpy().T
        title='Muscle activation at collocation points' 
        plotVSBounds(y,lwp,uwp,title)  
        # Muscle force at mesh points.
        lwp = lw['F'].to_numpy().T
        uwp = uw['F'].to_numpy().T
        y = w0['F'].to_numpy().T
        title='Muscle force at mesh points' 
        plotVSBounds(y,lwp,uwp,title)  
        # Muscle force at collocation points.
        lwp = lw['F'].to_numpy().T
        uwp = uw['F'].to_numpy().T
        y = w0['Fj'].to_numpy().T
        title='Muscle force at collocation points' 
        plotVSBounds(y,lwp,uwp,title)        
    # Joint position at mesh points.
    lwp = np.reshape(
        lw['Qsk'], (nJoints, N+1), order='F')
    uwp = np.reshape(
        uw['Qsk'], (nJoints, N+1), order='F')
    y = guessQsEnd
    title='Joint position at mesh points' 
    plotVSvaryingBounds(y,lwp,uwp,title)             
    # Joint position at collocation points.
    lwp = np.reshape(
        lw['Qsj'], (nJoints, d*N), order='F')
    uwp = np.reshape(
        uw['Qsj'], (nJoints, d*N), order='F')
    y = w0['Qsj'].to_numpy().T
    title='Joint position at collocation points' 
    plotVSvaryingBounds(y,lwp,uwp,title) 
    # Joint velocity at mesh points.
    lwp = lw['Qds'].to_numpy().T
    uwp = uw['Qds'].to_numpy().T
    y = guessQdsEnd
    title='Joint velocity at mesh points' 
    plotVSBounds(y,lwp,uwp,title) 
    # Joint velocity at collocation points.
    lwp = lw['Qds'].to_numpy().T
    uwp = uw['Qds'].to_numpy().T
    y = w0['Qdsj'].to_numpy().T
    title='Joint velocity at collocation points' 
    plotVSBounds(y,lwp,uwp,title) 
    if withArms:
        # Arm activation at mesh points.
        lwp = lw['ArmA'].to_numpy().T
        uwp = uw['ArmA'].to_numpy().T
        y = w0['ArmA'].to_numpy().T
        title='Arm activation at mesh points' 
        plotVSBounds(y,lwp,uwp,title) 
        # Arm activation at collocation points.
        lwp = lw['ArmA'].to_numpy().T
        uwp = uw['ArmA'].to_numpy().T
        y = w0['ArmAj'].to_numpy().T
        title='Arm activation at collocation points' 
        plotVSBounds(y,lwp,uwp,title)
    if withLumbarCoordinateActuators:
        # Lumbar activation at mesh points.
        lwp = lw['LumbarA'].to_numpy().T
        uwp = uw['LumbarA'].to_numpy().T
        y = w0['LumbarA'].to_numpy().T
        title='Lumbar activation at mesh points' 
        plotVSBounds(y,lwp,uwp,title) 
        # Lumbar activation at collocation points.
        lwp = lw['LumbarA'].to_numpy().T
        uwp = uw['LumbarA'].to_numpy().T
        y = w0['LumbarAj'].to_numpy().T
        title='Lumbar activation at collocation points' 
        plotVSBounds(y,lwp,uwp,title)
    # Controls.
    if torque_driven_model:
        # Coordinate excitation at mesh points.
        lwp = lw['CoordE'].to_numpy().T
        uwp = uw['CoordE'].to_numpy().T
        y = w0['CoordE'].to_numpy().T
        title='Coordinate excitation at mesh points' 
        plotVSBounds(y,lwp,uwp,title)
    else:
        # Muscle activation derivative at mesh points.
        lwp = lw['ADt'].to_numpy().T
        uwp = uw['ADt'].to_numpy().T
        y = w0['ADt'].to_numpy().T
        title='Muscle activation derivative at mesh points' 
        plotVSBounds(y,lwp,uwp,title) 
    if withArms:
        # Arm excitation at mesh points.
        lwp = lw['ArmE'].to_numpy().T
        uwp = uw['ArmE'].to_numpy().T
        y = w0['ArmE'].to_numpy().T
        title='Arm excitation at mesh points' 
        plotVSBounds(y,lwp,uwp,title)
    if withLumbarCoordinateActuators:
        # Lumbar excitation at mesh points.
        lwp = lw['LumbarE'].to_numpy().T
        uwp = uw['LumbarE'].to_numpy().T
        y = w0['LumbarE'].to_numpy().T
        title='Lumbar excitation at mesh points' 
        plotVSBounds(y,lwp,uwp,title)
    if not torque_driven_model:             
        # Muscle force derivative at mesh points.
        lwp = lw['FDt'].to_numpy().T
        uwp = uw['FDt'].to_numpy().T
        y = w0['FDt'].to_numpy().T
        title='Muscle force derivative at mesh points' 
        plotVSBounds(y,lwp,uwp,title)
    # Joint velocity derivative (acceleration) at mesh points.
    lwp = lw['Qdds'].to_numpy().T
    uwp = uw['Qdds'].to_numpy().T
    y = w0['Qdds'].to_numpy().T
    title='Joint velocity derivative (acceleration) at mesh points' 
    plotVSBounds(y,lwp,uwp,title)
    
def plotOptimalSolutionVSBounds(lw, uw, c_wopt, torque_driven_model=False):
    # States
    if torque_driven_model:
        # Coordinate activation at mesh points
        lwp = lw['CoordA'].to_numpy().T
        uwp = uw['CoordA'].to_numpy().T
        y = c_wopt['aCoord_opt']
        title='Coordinate activation at mesh points'            
        plotVSBounds(y,lwp,uwp,title)  
        # Coordinate activation at collocation points
        lwp = lw['CoordA'].to_numpy().T
        uwp = uw['CoordA'].to_numpy().T
        y = c_wopt['aCoord_col_opt']
        title='Coordinate activation at collocation points' 
        plotVSBounds(y,lwp,uwp,title)
    else:
        # Muscle activation at mesh points
        lwp = lw['A'].to_numpy().T
        uwp = uw['A'].to_numpy().T
        y = c_wopt['a_opt']
        title='Muscle activation at mesh points'            
        plotVSBounds(y,lwp,uwp,title)  
        # Muscle activation at collocation points
        lwp = lw['A'].to_numpy().T
        uwp = uw['A'].to_numpy().T
        y = c_wopt['a_col_opt']
        title='Muscle activation at collocation points' 
        plotVSBounds(y,lwp,uwp,title)  
        # Muscle force at mesh points
        lwp = lw['F'].to_numpy().T
        uwp = uw['F'].to_numpy().T
        y = c_wopt['nF_opt']
        title='Muscle force at mesh points' 
        plotVSBounds(y,lwp,uwp,title)  
        # Muscle force at collocation points
        lwp = lw['F'].to_numpy().T
        uwp = uw['F'].to_numpy().T
        y = c_wopt['nF_col_opt']
        title='Muscle force at collocation points' 
        plotVSBounds(y,lwp,uwp,title)
    # Joint position at mesh points
    lwp = lw['Qs'].to_numpy().T
    uwp = uw['Qs'].to_numpy().T
    y = c_wopt['Qs_opt']
    title='Joint position at mesh points' 
    plotVSBounds(y,lwp,uwp,title)             
    # Joint position at collocation points
    lwp = lw['Qs'].to_numpy().T
    uwp = uw['Qs'].to_numpy().T
    y = c_wopt['Qs_col_opt']
    title='Joint position at collocation points' 
    plotVSBounds(y,lwp,uwp,title) 
    # Joint velocity at mesh points
    lwp = lw['Qds'].to_numpy().T
    uwp = uw['Qds'].to_numpy().T
    y = c_wopt['Qds_opt']
    title='Joint velocity at mesh points' 
    plotVSBounds(y,lwp,uwp,title) 
    # Joint velocity at collocation points
    lwp = lw['Qds'].to_numpy().T
    uwp = uw['Qds'].to_numpy().T
    y = c_wopt['Qds_col_opt']
    title='Joint velocity at collocation points' 
    plotVSBounds(y,lwp,uwp,title)
    # Controls
    if torque_driven_model:
        # Muscle activation derivative at mesh points
        lwp = lw['CoordE'].to_numpy().T
        uwp = uw['CoordE'].to_numpy().T
        y = c_wopt['eCoord_opt']
        title='Coordinate excitation at mesh points' 
        plotVSBounds(y,lwp,uwp,title)
    else:
        # Muscle activation derivative at mesh points
        lwp = lw['ADt'].to_numpy().T
        uwp = uw['ADt'].to_numpy().T
        y = c_wopt['aDt_opt']
        title='Muscle activation derivative at mesh points' 
        plotVSBounds(y,lwp,uwp,title)
    # Slack controls
    if not torque_driven_model:
        # Muscle force derivative at collocation points
        lwp = lw['FDt'].to_numpy().T
        uwp = uw['FDt'].to_numpy().T
        y = c_wopt['nFDt_opt']
        title='Muscle force derivative at collocation points' 
        plotVSBounds(y,lwp,uwp,title)
    # Joint velocity derivative (acceleration) at collocation points
    lwp = lw['Qdds'].to_numpy().T
    uwp = uw['Qdds'].to_numpy().T
    y = c_wopt['Qdds_opt']
    title='Joint velocity derivative (acceleration) at collocation points' 
    plotVSBounds(y,lwp,uwp,title)