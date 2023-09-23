1) Identify the origin and frame of your force plates.
2) Identify a way to consistently place the checkerboard relative to the force plates. Black square at top left.
3) Measure from the checkerboard origin to force plate origin, expressed in checkerboard frame.

![Example](![forceTransform](https://github.com/stanfordnmbl/opencap-processing/assets/43877159/a37b0930-cdc9-47ad-b023-6d9c001c938f)


4) After running the example_integrate_forceplates.py, sanity check that forces align with kinematics using OpenSim.
![image](https://github.com/stanfordnmbl/opencap-processing/assets/43877159/4cea01ca-b89c-46d4-8c08-fd2f6d16907c)
a) load model file (.osim) that was downloaded from OpenCap into OpenSim (ensure there is not model offset by right clicking on it in the navigator and making Model Offset = 0 in all directions)
b) load inverse kinematics file downloaded from OpenCap
c) right click on the Motion in the Navigator, click 'Associate Motion Data,' and select the ground reaction forces (OpenCapData/MeasuredForces/<trial_name>/<trial_name>_syncd_forces.mot)

