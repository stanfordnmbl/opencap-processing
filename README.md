# OpenCap Processing

This repository enables the post-processing of human movement kinematics collected using [OpenCap](opencap.ai). You can run kinematic analyses, download multiple sessions using scripting, and run muscle-driven simulations to estimate kinetics.

## Publication
More information is available in our [paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462): <br> <br>
Uhlrich SD*, Falisse A*, Kidzinski L*, Ko M, Chaudhari AS, Hicks JL, Delp SL, 2022. OpenCap: Human movement dynamics from smartphone videos. PLoS Comput Biol 19(10): e1011462. https://doi.org/10.1371/journal.pcbi.1011462. *contributed equally <br> <br>
Archived code base corresponding to publication: https://zenodo.org/record/7419973

## Install requirements
### General
1. Install [Anaconda](https://www.anaconda.com/)
1. Open Anaconda prompt
2. Create environment (python 3.11 recommended): `conda create -n opencap-processing python=3.11`
3. Activate environment: `conda activate opencap-processing`
4. Install OpenSim: `conda install -c opensim-org opensim=4.5=py311np123`
    - Test that OpenSim was successfully installed:
        - Start python: `python`
        - Import OpenSim: `import opensim`
            - If you don't get any error message at this point, you should be good to go.
        - You can also double check which version you installed : `opensim.GetVersion()`
        - Exit python: `quit()`
    - Visit this [webpage](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53116061/Conda+Package) for more details about the OpenSim conda package.
5. (Optional): Install an IDE such as Spyder: `conda install spyder`
6. Clone the repository to your machine: 
    - Navigate to the directory where you want to download the code: eg. `cd Documents`. Make sure there are no spaces in this path.
    - Clone the repository: `git clone https://github.com/stanfordnmbl/opencap-processing.git`
    - Navigate to the directory: `cd opencap-processing`
7. Install required packages: `python -m pip install -r requirements.txt`
8. Run `python createAuthenticationEnvFile.py`
    - An environment variable (`.env` file) will be saved after authenticating.    
    
### Muscle-driven simulations
1. **Windows only**: Install [Visual Studio](https://visualstudio.microsoft.com/downloads/)
    - The Community variant is sufficient and is free for everyone.
    - During the installation, select the *workload Desktop Development with C++*.
    - The code was tested with the 2017, 2019, and 2022 Community editions.
2. **Linux only**: Install OpenBLAS libraries
    - `sudo apt-get install libopenblas-base`

    
## Examples
- Run `example.py` for examples of how to run kinematic analyses
- Run `example_kinetics.py` for examples of how to generate muscle-driven simulations
- Moco
    - The [Moco folder](https://github.com/stanfordnmbl/opencap-processing/tree/main/Moco) contains examples for generating muscle-driven simulations using [OpenSim Moco](https://opensim-org.github.io/opensim-moco-site/). 

## Download OpenCap data

### Using Colab
- Open `batchDownload.ipynb` in Colab and follow the instructions
    - You do not need to follow the install requirements above.

### Locally
- Follow the install requirements above
- Open `batchDownload.py` and follow the instructions
