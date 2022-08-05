'''
    ---------------------------------------------------------------------------
    OpenCap processing: utilsProcessing.py
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
from scipy import signal

def lowPassFilter(time, data, lowpass_cutoff_frequency, order=4):
    
    fs = 1/np.round(np.mean(np.diff(time)),16)
    wn = lowpass_cutoff_frequency/(fs/2)
    sos = signal.butter(order/2, wn, btype='low', output='sos')
    dataFilt = signal.sosfiltfilt(sos, data, axis=0)

    return dataFilt
