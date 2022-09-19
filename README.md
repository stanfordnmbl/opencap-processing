
# Install requirements
- Open Anaconda prompt
- Create environment (python 3.9 recommended): `conda create -n opencap-processing python=3.9`
- Activate environment: `activate opencap-processing`
- Install required packages: `python -m pip install -r requirements.txt`
    - (Optional): Install an IDE such as Spyder: `conda install spyder` 
- Install opensim: `conda install -c opensim-org opensim`
    - Visit this [webpage](https://simtk-confluence.stanford.edu:8443/display/OpenSim/Conda+Package) for more details about the opensim conda package
    
## Install requirements to run muscle-driven simulations
- Install [CMake](https://cmake.org/download/).
    - **Windows**: Add CMake to system path. During the installation, select *Add CMake to the system PATH for all users*.
- **Windows**: Install [Visual Studio](https://visualstudio.microsoft.com/downloads/).
    - The Community variant is sufficient and is free for everyone.
    - During the installation, select the workload Desktop Development with C++.
    - The code was tested with the 2017, 2019, and 2022 Community editions.
    
# Examples
- Run `example.py` for some kinematic analyses.
- Run `example_kinetics.py` to generate muscle-driven simulations.
