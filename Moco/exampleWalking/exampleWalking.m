% -------------------------------------------------------------------------- %
% OpenSim Moco: exampleWalking.m                                             %
% -------------------------------------------------------------------------- %
% Copyright (c) 2022 Stanford University and the Authors                     %
%                                                                            %
% Author(s): Carmichael Ong, Antoine Falisse, Scott Uhlrich                  %
%                                                                            %
% Licensed under the Apache License, Version 2.0 (the "License"); you may    %
% not use this file except in compliance with the License. You may obtain a  %
% copy of the License at http://www.apache.org/licenses/LICENSE-2.0          %
%                                                                            %
% Unless required by applicable law or agreed to in writing, software        %
% distributed under the License is distributed on an "AS IS" BASIS,          %
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   %
% See the License for the specific language governing permissions and        %
% limitations under the License.                                             %
% -------------------------------------------------------------------------- %

% Example using OpenSim Moco to track OpenCap data of walking. This example
% solves in about 5 hours on a computer with 8 cores.
%
% Model
% -----
% This script uses the same base model from OpenCap and then adjusts the
% model to use a muscle class better suited for use with Moco
% (DeGrooteFregly2016), to increase the optimal force of
% CoordinateActuators that actuate the lumbar and upper limb joints, and to
% weld the toe (mtp) joints.
%
% Data
% ----
% Data are from a processed OpenCap trial of walking and contain coordinate
% values after inverse kinematics.

clear;

% Load the Moco libraries
import org.opensim.modeling.*;

% ---------------------------------------------------------------------------
% Set up a coordinate tracking problem where the goal is to minimize the
% difference between provided and simulated coordinate values and speeds, 
% as well as to minimize an effort cost (squared controls). Endpoint
% constraints enforce periodicity of the coordinate values (except for
% pelvis tx) and speeds, coordinate actuator controls, and muscle 
% activations.


% Define the optimal control problem
% ==================================
mocoTrackName = 'walking1';
track = MocoTrack();
track.setName(mocoTrackName);

% Set the weights for the terms in the objective function. The values below were
% obtained by trial and error.
controlEffortWeight = 0.1;
stateTrackingWeight = 1;

% Set other parameters, including the initial and final times of the
% simulation, the mesh interval, the relative size of the state bounds, and
% the optimal force for the lumbar and upper limb coordinate actuators.
initialTime = 0.07;
finalTime = 1.45;
meshInterval = 0.05;
fractionExtraBoundSize = 0.1;
coordinateActuatorsOptimalForce = 250;

% Reference data for tracking problem
tableProcessor = TableProcessor('walking1_videoAndMocap.mot');
tableProcessor.append(TabOpLowPassFilter(6));
tableProcessor.append(TabOpUseAbsoluteStateNames());

% Load the model and increase the optimal force for the coordinate
% actuators that control the lumbar and upper limbs.
inModel = Model('../models/LaiArnoldModified2017_contacts.osim');
setCoordinateActuatorsOptimalForce(inModel, coordinateActuatorsOptimalForce);

% Process the model to weld the toe (mtp) joints, replace the muscles in
% the model to use the DeGrooteFregly2016 muscles, and to use the implicit
% mode for the muscles.
modelProcessor = ModelProcessor(inModel);
jointsToWeld = StdVectorString();
jointsToWeld.add('mtp_r');
jointsToWeld.add('mtp_l');
modelProcessor.append(ModOpReplaceJointsWithWelds(jointsToWeld));
modelProcessor.append(ModOpReplaceMusclesWithDeGrooteFregly2016());
modelProcessor.append(ModOpTendonComplianceDynamicsModeDGF('implicit')); 

% Set the settings for the MocoTrack problem.
track.setModel(modelProcessor);
track.setStatesReference(tableProcessor);
track.set_states_global_tracking_weight(stateTrackingWeight);
track.set_allow_unused_references(true);
track.set_track_reference_position_derivatives(true);
track.set_apply_tracked_states_to_guess(true);
track.set_initial_time(initialTime);
track.set_final_time(finalTime);
track.set_mesh_interval(meshInterval);
study = track.initialize();
problem = study.updProblem();


% Goals
% =====

% Tracking
% --------
% Set different tracking weights for states (weights for states not 
% explicitly set here have a default value of 1.0)
stateTrackingGoal = MocoStateTrackingGoal.safeDownCast(problem.updGoal('state_tracking'));
stateTrackingGoal.setWeightForState('/jointset/ground_pelvis/pelvis_tilt/value', 10.0);
stateTrackingGoal.setWeightForState('/jointset/ground_pelvis/pelvis_tilt/speed', 10.0);
stateTrackingGoal.setWeightForState('/jointset/back/lumbar_extension/value', 10.0);
stateTrackingGoal.setWeightForState('/jointset/back/lumbar_extension/speed', 10.0);
stateTrackingGoal.setWeightForState('/jointset/hip_r/hip_flexion_r/value', 10.0);
stateTrackingGoal.setWeightForState('/jointset/hip_r/hip_flexion_r/speed', 10.0);
stateTrackingGoal.setWeightForState('/jointset/hip_l/hip_flexion_l/value', 10.0);
stateTrackingGoal.setWeightForState('/jointset/hip_l/hip_flexion_l/speed', 10.0);
stateTrackingGoal.setWeightForState('/jointset/walker_knee_r/knee_angle_r/value', 10.0);
stateTrackingGoal.setWeightForState('/jointset/walker_knee_r/knee_angle_r/speed', 10.0);
stateTrackingGoal.setWeightForState('/jointset/walker_knee_l/knee_angle_l/value', 10.0);
stateTrackingGoal.setWeightForState('/jointset/walker_knee_l/knee_angle_l/speed', 10.0);
stateTrackingGoal.setWeightForState('/jointset/ankle_r/ankle_angle_r/speed', 10.0);
stateTrackingGoal.setWeightForState('/jointset/ankle_r/ankle_angle_r/value', 10.0);
stateTrackingGoal.setWeightForState('/jointset/ankle_l/ankle_angle_l/speed', 10.0);
stateTrackingGoal.setWeightForState('/jointset/ankle_l/ankle_angle_l/value', 10.0);

% Periodicity
% -----------
% This goal enforces periodicity (i.e., enforce that certain values are
% the same at the beginning of the simulation as the end of the
% simulation).
periodicityGoal = MocoPeriodicityGoal('periodicityGoal');
problem.addGoal(periodicityGoal);
model = modelProcessor.process();
model.initSystem();

% Periodic coordinate values (except for pelvis_tx) and speeds.
for i = 1:model.getNumStateVariables()
    currentStateName = string(model.getStateVariableNames().getitem(i-1));
    if startsWith(currentStateName , '/jointset')
        if (~contains(currentStateName, 'pelvis_tx/value'))
            periodicityGoal.addStatePair(MocoPeriodicityGoalPair(currentStateName));
        end
    end
end

% Periodic muscle activations.
for i = 1:model.getNumStateVariables()
    currentStateName = string(model.getStateVariableNames().getitem(i-1));
    if endsWith(currentStateName,'/activation')
        periodicityGoal.addStatePair(MocoPeriodicityGoalPair(currentStateName));
    end
end

% Periodiic coordinate actuator (lumbar and upper limb) controls.
modelActuators = model.getActuators();
for i = 1:modelActuators.getSize()
    thisActuator = modelActuators.get(i-1);
    if strcmp(thisActuator.getConcreteClassName, 'CoordinateActuator')
        thisPair = MocoPeriodicityGoalPair(thisActuator.getAbsolutePathString());
        periodicityGoal.addControlPair(thisPair);
    end
end

% Effort
% ------
% Get a reference to the MocoControlGoal that is added to every MocoTrack
% problem by default and change the weight
effort = MocoControlGoal.safeDownCast(problem.updGoal('control_effort'));
effort.setWeight(controlEffortWeight);


% Bounds
% ======
% Set the bounds for all tracked states based on the range of the data.

% Load the tracked_states file into a table to find the range of the
% coordinate values in the data.
trackedStatesFile = [mocoTrackName, '_tracked_states.sto'];
trackedStatesTable = TimeSeriesTable(trackedStatesFile);
trackedStatesTable.trimFrom(initialTime);
trackedStatesTable.trimTo(finalTime);

% Use the helper function to constrain the states in the vector based on
% the range of the data in the table.
constrainBoundsAllTrackedStates(problem, trackedStatesTable, ...
    fractionExtraBoundSize);

% Solve the problem
% =================
% Add a term to the solver that minimizes the implicit auxiliary
% derivatives that are used with the implicit mode of muscles.
solver = MocoCasADiSolver.safeDownCast(study.updSolver());
solver.set_minimize_implicit_auxiliary_derivatives(true);
solver.set_implicit_auxiliary_derivatives_weight(1e-6);

% Solve the problem and save the results.
solution = study.solve();
solution.write(strcat(mocoTrackName, '.sto'));

% Extract ground reaction forces
% ==============================
% Extract ground reaction forces for downstream analysis. Add the contact
% force elements to vectors, then use Moco's
% createExternalLoadsTableForGait() function.
contact_r = StdVectorString();
contact_l = StdVectorString();
contact_r.add('/forceset/SmoothSphereHalfSpaceForce_s1_r');
contact_r.add('/forceset/SmoothSphereHalfSpaceForce_s2_r');
contact_r.add('/forceset/SmoothSphereHalfSpaceForce_s3_r');
contact_r.add('/forceset/SmoothSphereHalfSpaceForce_s4_r');
contact_r.add('/forceset/SmoothSphereHalfSpaceForce_s5_r');
contact_r.add('/forceset/SmoothSphereHalfSpaceForce_s6_r');
contact_l.add('/forceset/SmoothSphereHalfSpaceForce_s1_l');
contact_l.add('/forceset/SmoothSphereHalfSpaceForce_s2_l');
contact_l.add('/forceset/SmoothSphereHalfSpaceForce_s3_l');
contact_l.add('/forceset/SmoothSphereHalfSpaceForce_s4_l');
contact_l.add('/forceset/SmoothSphereHalfSpaceForce_s5_l');
contact_l.add('/forceset/SmoothSphereHalfSpaceForce_s6_l');

model = modelProcessor.process();
externalForcesTableFlat = ...
    opensimMoco.createExternalLoadsTableForGait(model, ...
                                 solution.exportToStatesTrajectory(model), ...
                                 contact_r, contact_l);
STOFileAdapter.write(externalForcesTableFlat, strcat(mocoTrackName, '_grfs.sto'));

%% Helper functions

% Set each state's bounds based on a fraction (fractionExtraBoundSize) of
% the range of the state's value from the tracked data. The bounds are
% set as the following:
%   - Lower bound: (minimum value) - fractionExtraBoundSize * (range of value)
%   - Upper bound: (maximum value) + fractionExtraBoundSize * (range of value)
function constrainBoundsAllTrackedStates(problem, trackedStatesTable, ...
    fractionExtraBoundSize)
    
    % Get all of the state names from the table column labels but omit
    % the toe joint since it was locked in the model above. Store the
    % state names in a vector to pass into the function that constrains the 
    % bounds.
    colLabelsStdVec = trackedStatesTable.getColumnLabels();
    colLabelsCell = cell(1, colLabelsStdVec.size());
    for i = 1:colLabelsStdVec.size()
        thisColLabel = char(colLabelsStdVec.get(i-1));
        if ~contains(thisColLabel, 'mtp_angle')
            colLabelsCell{i} = thisColLabel;
        end
    end
    colLabelsCell = colLabelsCell(~cellfun('isempty', colLabelsCell));
    
    for i = 1:numel(colLabelsCell)
        thisStatePath = colLabelsCell{i};
        stateColumn = ...
            trackedStatesTable.getDependentColumn(thisStatePath).getAsMat();
        thisColMin = min(stateColumn);
        thisColMax = max(stateColumn);
        thisColRange = thisColMax - thisColMin;
        extraBoundSize = thisColRange * fractionExtraBoundSize;
        
        thisBounds = [thisColMin - extraBoundSize, thisColMax + extraBoundSize];
        problem.setStateInfo(thisStatePath, thisBounds, [], []);
    end
    
end

% Set the optimal force of all CoordinateActuator's in the model to a given
% value (optimalForce).
function setCoordinateActuatorsOptimalForce(model, optimalForce)
    % import inside of the function needed for downcast step
    import org.opensim.modeling.*;

    modelActuators = model.getActuators();
    for i = 1:modelActuators.getSize()
        thisActuator = modelActuators.get(i-1);
        if strcmp(thisActuator.getConcreteClassName(), 'CoordinateActuator')
            thisCoordinateActuator = CoordinateActuator.safeDownCast(thisActuator);
            thisCoordinateActuator.setOptimalForce(optimalForce);
        end
    end

end