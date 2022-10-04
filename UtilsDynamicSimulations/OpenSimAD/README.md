# Muscle-driven simulations with OpenSimAD

The examples we provide to generate muscle-driven simulations use OpenSimAD, which is a custom version of OpenSim that supports automatic differentiation (AD). AD is an alternative to finite differences to compute derivatives, and is supposedly faster. You can find more details about OpenSimAD and some benchmarking against finite differences in [this publication](https://journals.plos.org/plosone/article/comments?id=10.1371/journal.pone.0217730). Please keep in mind that OpenSimAD does not support all features of OpenSim, you should therefore carefully verify what you are doing should you diverge from the provided examples (eg, if you use a different musculoskeletal model). We will contribute examples to generate muscle-driven simulations using Moco in the near future.

OpenSimAD requires compiling C++ and C code. Everything is automated, but please follow the [specific install requirements](https://github.com/stanfordnmbl/opencap-processing#install-requirements-to-run-muscle-driven-simulations) to make sure you have everything you need (CMake and compiler).

### Overview pipeline muscle-driven simulations
1. **Process inputs**
  - [Download the model and the motion file](https://github.com/stanfordnmbl/opencap-processing/blob/main/UtilsDynamicSimulations/OpenSimAD/utilsOpenSimAD.py#L1912) with the coordinate values estimated from videos.
  - [Adjust the wrapping surfaces](https://github.com/stanfordnmbl/opencap-processing/blob/main/UtilsDynamicSimulations/OpenSimAD/utilsOpenSimAD.py#L1917) of the model to enforce meaningful moment arms (ie, address known bug of wrapping surfaces).
  - [Add contact spheres](https://github.com/stanfordnmbl/opencap-processing/blob/main/UtilsDynamicSimulations/OpenSimAD/utilsOpenSimAD.py#L1919) to the musculoskeletal model to model foot-ground interactions.
  - [Generate differentiable external function](https://github.com/stanfordnmbl/opencap-processing/blob/main/UtilsDynamicSimulations/OpenSimAD/utilsOpenSimAD.py#L1921) to leverage AD when solving the optimal control problem.
    - More details about this process in [this publication](https://journals.plos.org/plosone/article/comments?id=10.1371/journal.pone.0217730) and [this repository](https://github.com/antoinefalisse/opensimAD).
2. **Fit polynomials to approximate muscle-tendon lenghts and velocities, and moment arms.**
  - We use polynomial approximations of coordinates values to estimate muscle-tendon lenghts and velocities, and moment arms. We [fit the polynomial coefficients](https://github.com/stanfordnmbl/opencap-processing/blob/main/UtilsDynamicSimulations/OpenSimAD/mainOpenSimAD.py#L541) before solving the optimal control problem. Using polynomial approximations has the advantage of speeding up evaluations of muscle-tendon lenghts and velocities, and moment arms.
3. **Solve optimal control / trajectory optimization problem**
- We generate [muscle-driven tracking simulations](https://github.com/stanfordnmbl/opencap-processing/blob/main/UtilsDynamicSimulations/OpenSimAD/mainOpenSimAD.py#L999) of joint kinematics. The general idea is to solve for the model controls that will drive the musculoskeletal model to closely track the measured kinematics while satisfying the dynamic equations describing muscle and skeletal dynamics and minimizing muscle effort. We use direct collocation methods to solved this problem, and leverage AD through [CasADi](https://web.casadi.org/).
4. **Process results**
- From the simulations, we can extract dynamic variables like muscle forces, joint moments, ground reaction forces, or joint contact forces.

### Overview outputs
- If your problem converges, you should get a few files under OpenSimData/Dynamics:
  - forces_<trial_name>.mot
    - Muscle forces and non muscle-driven joint torques (eg, reserve actuators).
  - GRF_resultant_<trial_name>.mot
    - Resultant ground reaction forces and moments.
  - GRF_<trial_name>.mot
    - Ground reaction forces and moments (per contact sphere).
  - kinematics_activations_<trial_name>.mot
    - Joint kinematics and muscle activations.
  - kinetics_<trial_name>.mot
    - Joint moments.
  - optimaltrajectories.npy
    - Dictionnary with more results:
      - time: discretized time vector.
      - coordinate_values_toTrack: reference coordinate values estimated from videos.
      - coordinate_values: coordinate values resulting from the dynamic simulation.
      - coordinate_speeds_toTrack: reference coordinate speeds estimated from videos.
      - coordinate_speeds: coordinate speeds resulting from the dynamic simulation.
      - coordinate_accelerations_toTrack: reference coordinate accelerations estimated from videos.
      - coordinate_accelerations: coordinate accelerations resulting from the dynamic simulation.
      - torques: joint torques/moments from the dynamic simulation.
      - torques_BWht: joint torques/moments normalized by body weight times height from the dynamic simulation.
      - GRF: resultant ground reaction forces from the dynamic simulation.
      - GRF_BW: resultant ground reaction forces normalized by body weight from the dynamic simulation.
      - GRM: resultant ground reaction moments from the dynamic simulation.
      - GRM_BWht: resultant ground reaction moments normalized by body weight times height from the dynamic simulation.
      - muscle_activations: muscle activations from the dynamic simulation.
      - passive_muscle_torques: passive torque contribution of the muscles from the dynamic simulation.
      - active_muscle_torques: active torque contribution of the muscles from the dynamic simulation.
      - passive_limit_torques: torque contributions from limit torques from the dynamic simulation.
      - KAM: knee adduction moments from the dynamic simulation.
      - KAM_BWht: knee adduction moments normalized by body weight times height from the dynamic simulation.
      - MCF: medial knee contact forces from the dynamic simulation.
      - MCF_BW: medial knee contact forces normalized by body weight from the dynamic simulation.
      - coordinates: coordinate names.
      - rotationalCoordinates: rotational coordinate names.
      - GRF_labels: labels ground reaction forces
      - muscles: muscle names.
      - muscle_driven_joints: muscle-driven coordinate names.
      - limit_torques_joints : names of coordinates with limit torques.
      - KAM_labels: labels knee adduction moments.
      - MCF_labels: labels medial knee contact fprces.
      - iter_count: number of iterations the problem took to converge.

### Overview files
- `boundsOpenSimAD.py`: script describing the bounds of the problem variables.
- `functionCasADiOpenSimAD.py`: various helper CasADi functions.
- `initialGuessOpenSimAD.py`: script describing the initial guess of the problem variables.
- `mainOpenSimAD.py`: main script formulating and solving the problem.
- `muscleDataOpenSimAD.py`: various helper functions related to muscle models.
- `muscleModelOpenSimAD.py`: implementation of the [DeGrooteFregly](https://pubmed.ncbi.nlm.nih.gov/27001399/) muscle model.
- `plotsOpenSimAD.py`: helper plots for intermediate visualization.
- `polynomialsOpenSimAD.py`: script to fit polynomial coefficients.
- `settingsOpenSimAD.py`: settings for the problem with pre-defined settings for simulating different activities.
- `utilsOpenSimAD.py`: various utilities for OpenSimAD.

### Food for thoughts

Dynamic simulations of human movement require solving complex optimal control problems. **It is a tedious task with no guarantee of success.** Even if the problem converges (*optimal solution found*), you should always verify that the results are biomechanically meaningful. It is possible that the problem satisfied all constraints but did not converge to the expected solution. You might want to play with the settings (eg, weights of the different terms in the cost function), constraints, and cost function terms to generate simulations that make sense for the particular activity you are interested in.
