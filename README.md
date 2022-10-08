This repository enables the postprocessing of results captured from [OpenCap](opencap.ai). You can run kinematic analyses, download multiple sessions using scripting, and run muscle-driven simulations to estimate kinetics.

# Publication
## Publication
More information is available in our [preprint](https://www.biorxiv.org/content/10.1101/2022.07.07.499061v1): <br> <br>
Uhlrich SD*, Falisse A*, Kidzinski L*, Ko M, Chaudhari AS, Hicks JL, Delp SL, 2022. OpenCap: 3D human movement dynamics from smartphone videos. _biorxiv_. https://doi.org/10.1101/2022.07.07.499061. *contributed equally

# Install requirements
## General
1. Open Anaconda prompt.
2. Create environment (python 3.9 recommended): `conda create -n opencap-processing python=3.9`.
3. Activate environment: `conda activate opencap-processing`.
4. Install OpenSim: `conda install -c opensim-org opensim=4.4=py39np120`.
    - Visit this [webpage](https://simtk-confluence.stanford.edu:8443/display/OpenSim/Conda+Package) for more details about the OpenSim conda package.
5. Navigate to the directory where this repository is cloned and install required packages: `python -m pip install -r requirements.txt`.
    - (Optional): Install an IDE such as Spyder: `conda install spyder`.
    
## Muscle-driven simulations
1. Install [CMake](https://cmake.org/download/)
    - **Windows only**: Add CMake to system path. During the installation, select *Add CMake to the system PATH for all users*
2. **Windows only**: Install [Visual Studio](https://visualstudio.microsoft.com/downloads/)
    - The Community variant is sufficient and is free for everyone.
    - During the installation, select the *workload Desktop Development with C++*.
    - The code was tested with the 2017, 2019, and 2022 Community editions.
    
# Examples
- Run `example.py` for examples of how to run kinematic analyses.
- Run `example_kinetics.py` for examples of how to generate muscle-driven simulations.

# Download OpenCap data

## Using Colab
- Open `batchDownload.ipynb` in Colab and follow the instructions.
    - You do not need to follow the install requirements above.

## Locally
- Follow the install requirements above.
- (Optional): Run `createAuthenticationEnvFile.py`.
    - An environment variable (`.env` file) will be saved after authenticating. You can proceed without this, but you will be required to login every time you run a script.
- Open `batchDownload.py` and follow the instructions.
