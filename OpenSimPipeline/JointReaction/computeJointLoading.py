# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 15:20:10 2021

@author: suhlr
"""

# Computes KAM from IK, ID, and GRFs

import opensim
import os
import numpy as np
import copy
import glob


# %% This was to get to xml path, but opensim didn't like loading relative xml paths
def getBaseDir():
    functionAbsPath = os.path.abspath(__file__)
    head, tail = os.path.split(functionAbsPath)
    head,tail = os.path.split(head)
    upString = ''
    while tail != 'mobilecap':
        head, tail = os.path.split(head)
        upString += '../'
    baseRelPath = upString
    
    return baseRelPath

#%% 
def computeKAM(outputDir,modelPath,IDPath,IKPath,GRFPath,grfType, Qds=[]):

    baseRelPath = getBaseDir()
    jointReactionXmlPath = os.path.join(baseRelPath,'opensimPipeline\JointReaction\Setup_JointReaction.xml')

    
    # %% Editable variables
    statesInDegrees = True # defaults to reading input file, but if not in header, uses this
    removeSpheres = True # Delete spheres as force elements
    # grfType = 'sphere' # Options: 'sphere','sphereResultant','experimental'
    
    # %%
    
    # Load model
    opensim.Logger.setLevelString('error')
    model = opensim.Model(modelPath)
    
    # Remove all actuators and add coordinate actuators                                            
    forceSet = model.getForceSet()
    
    i = 0
    while i < forceSet.getSize():
        if removeSpheres:
            forceSet.remove(i)
        else:
            if 'SmoothSphere' not in forceSet.get(i).getConcreteClassName():
                forceSet.remove(i)
            else:
                i+=1
      
    coords = model.getCoordinateSet()
    nCoords = coords.getSize()
    coordNames = [coords.get(i).getName() for i in range(nCoords)]
    
    actuatorNames = []
    for coord in coords:
        newActuator = opensim.CoordinateActuator(coord.getName())
        newActuator.setName(coord.getName() + '_actuator')
        actuatorNames.append(coord.getName() + '_actuator')
        newActuator.set_min_control(-np.inf)
        newActuator.set_max_control(np.inf)
        newActuator.set_optimal_force(1)
        model.addForce(newActuator)
        
        # Add Prescribed Controllers for Coordinate Actuators (Needed for JRA to
        # work properly)
        
        # Construct constant function
        constFxn = opensim.Constant(0) 
        constFxn.setName(coord.getName() + '_constFxn') 
         
        # Construct prescribed controller
        pController = opensim.PrescribedController() 
        pController.setName(coord.getName() + '_controller') 
        pController.addActuator(newActuator) 
        pController.prescribeControlForActuator(0,constFxn) # attach the function to the controller
        model.addController(pController) 
        
    actuators = model.getActuators()
    controllerSet = model.getControllerSet()
    
    # Load ID moments
    idTable = opensim.TimeSeriesTable(IDPath)
    idNames = idTable.getColumnLabels()
    idTime = idTable.getIndependentColumn()
    
    # Load kinematic states, compute speeds
    stateTable = opensim.TimeSeriesTable(IKPath)
    stateNames = stateTable.getColumnLabels()
    stateTime = stateTable.getIndependentColumn()
    try:
        inDegrees = stateTable.getTableMetaDataAsString('inDegrees') == 'yes'
    except:
        inDegrees = statesInDegrees
        print('using statesInDegrees variable, which says statesInDegrees is ' + str(statesInDegrees))
    q = np.zeros((len(stateTime),nCoords))
    dt = stateTime[1] - stateTime[0]    
    if len(Qds) > 0:
        qd_t = np.zeros((len(stateTime),nCoords))
        
    for col in stateNames:
        if 'activation' in col:
            stateTable.removeColumn(col)
        else:
            coordCol = coordNames.index(col)
            for t in range(len(stateTime)):
                qTemp = np.asarray(stateTable.getDependentColumn(col)[t])
                
                if coords.get(col).getMotionType() == 1 and inDegrees: # rotation
                    qTemp = np.deg2rad(qTemp)
                q[t,coordCol] = copy.deepcopy(qTemp)
            if len(Qds) > 0:
                # get index
                idx_col = stateNames.index(col)
                qd_t[:,coordCol] = Qds[:, idx_col]
    # Add option to pass q_dot as argument.
    if not len(Qds) > 0:
        qd = np.diff(q, axis=0, prepend=np.reshape(q[0,:],(1,nCoords))) / dt
    else:
        qd = qd_t
    # qdd = np.diff(qd, axis=0, prepend=np.reshape(qd[0,:],(1,nCoords))) / dt # just for testing    
        
    # Load and apply GRFs
    dataSource = opensim.Storage(GRFPath) 
    
    if grfType == 'sphere':
        appliedToBody = ['calcn','calcn','calcn','calcn','toes','toes']
        for leg in ['r','l']:
            for i in range(len(appliedToBody)):
                newForce = opensim.ExternalForce()
                newForce.setName('sphere' + leg + str(i+1)) 
                newForce.set_applied_to_body(appliedToBody[i] + '_' + leg) 
                newForce.set_force_expressed_in_body('ground') 
                newForce.set_point_expressed_in_body('ground')
                newForce.set_force_identifier('ground_force_s' + str(i+1) + '_' + leg + '_v')
                newForce.set_torque_identifier('ground_torque_s' + str(i+1) + '_' + leg + '_') ;
                newForce.set_point_identifier('ground_force_s' + str(i+1) + '_' + leg + '_p') ;
                newForce.setDataSource(dataSource) ;
                if removeSpheres:
                    model.addForce(newForce) ;
                elif i==0:
                    print('GRFs were not applied b/c sphere contacts were used.')
    elif grfType == 'sphereResultant':
        appliedToBody = ['calcn']
        for leg in ['r','l']:
            for i in range(len(appliedToBody)):
                newForce = opensim.ExternalForce()
                newForce.setName('GRF' + '_' + leg + '_' + str(i)) 
                newForce.set_applied_to_body(appliedToBody[i] + '_' + leg.lower()) 
                newForce.set_force_expressed_in_body('ground') 
                newForce.set_point_expressed_in_body('ground')
                newForce.set_force_identifier('ground_force_' + leg + '_v')
                newForce.set_torque_identifier('ground_torque_' + leg + '_') ;
                newForce.set_point_identifier('ground_force_' + leg + '_p') ;
                newForce.setDataSource(dataSource) ;
                model.addForce(newForce) ; #tested and these make a difference
    elif grfType == 'experimental':
        appliedToBody = ['calcn']
        for leg in ['R','L']:
            for i in range(len(appliedToBody)):
                newForce = opensim.ExternalForce()
                newForce.setName('GRF' + '_' + leg + '_' + str(i)) 
                newForce.set_applied_to_body(appliedToBody[i] + '_' + leg.lower()) 
                newForce.set_force_expressed_in_body('ground') 
                newForce.set_point_expressed_in_body('ground')
                newForce.set_force_identifier(leg + '_ground_force_v')
                newForce.set_torque_identifier(leg + '_ground_torque_') ;
                newForce.set_point_identifier(leg + '_ground_force_p') ;
                newForce.setDataSource(dataSource) ;
                model.addForce(newForce) ; #tested and these make a difference
    
    # initSystem - done editing model
    state = model.initSystem()
    
    # Create state Y map
    yNames = opensim.createStateVariableNamesInSystemOrder(model)
    systemPositionInds = []
    systemVelocityInds = []
    stateNameList = []
    for stateName in coordNames:
        posIdx =np.squeeze(np.argwhere([stateName + '/value' in y for y in yNames]))
        velIdx =np.squeeze(np.argwhere([stateName + '/speed' in y for y in yNames])) 
        if posIdx.size>0:  
            systemPositionInds.append(posIdx)
            systemVelocityInds.append(velIdx)
            stateNameList.append(stateName)
    
    # Create JRA reporter
    jointReaction = opensim.JointReaction(jointReactionXmlPath)
    model.addAnalysis(jointReaction) ;
    jointReaction.setModel(model) ;
    jointReaction.printToXML(os.path.join(outputDir, 'JrxnSetup.xml')) ;    
    
    # Loop over time
    controls = opensim.Vector(nCoords,0) ;
    for iTime in range(len(stateTime)):
        thisTime = stateTime[iTime]
    
        if thisTime <= idTime[-1]:
            idRow = idTable.getNearestRowIndexForTime(thisTime)               
            # Set time
            state.setTime(thisTime)   
                
            # Set state, velocity, actuator controls
            yVec = np.zeros((state.getNY())).tolist()
            for iCoord, coord in enumerate(coords):
                if '_beta' not in coord.getName():
                    
    
                    # Loop thru states to set speeds and vels
                    yVec[systemPositionInds[iCoord]] = q[iTime,iCoord]
                    yVec[systemVelocityInds[iCoord]] = qd[iTime,iCoord]
                              
                    # Old/slow setting of speeds and velocity
                    # coord.setValue(state,q[iTime,iCoord])
                    # coord.setSpeedValue(state,qd[iTime,iCoord])
                    
                    if coord.getMotionType() == 1: # rotation
                        suffix = '_moment'
                    elif coord.getMotionType() == 2: # translation
                        suffix = '_force'
                        
                    # Set prescribed controller constant value to control value. Controls
                    # don't live through joint reaction analysis for some reason.
                    thisController = opensim.PrescribedController.safeDownCast(controllerSet.get(coord.getName() + '_controller')) 
                    thisConstFxn = opensim.Constant.safeDownCast(thisController.get_ControlFunctions(0).get(0))
                    thisConstFxn.setValue(idTable.getDependentColumn(coord.getName()+suffix)[idRow])
            # Setting controls this way is redundant, but necessary if want to do a force reporter
            # in future
                    controls.set(iCoord, idTable.getDependentColumn(coord.getName()+suffix)[idRow])
            
            state.setY(opensim.Vector(yVec))
            model.realizeVelocity(state)
                
            model.setControls(state,controls) # tested and commenting didn't impact JRA
            
            # Realize acceleration
            model.realizeAcceleration(state)
            
        # Compute JRA
        if iTime == 0:
            jointReaction.begin(state) 
        else:
            jointReaction.step(state,iTime) 
        if iTime == len(stateTime)-1 or thisTime >=idTime[-1]:
            jointReaction.end(state)
    
    # Finish time loop and output
    if not removeSpheres:
        grfType = 'spheresUsed_noGRFsApplied'
    outFileBase = 'results_JRA'
    jointReaction.printResults(outFileBase,outputDir,-1,'.sto')
    
    # Load and get KAM
    outFilePath = glob.glob(os.path.join(outputDir,outFileBase + '_JointReactionAnalysis_ReactionLoads.sto'))[0] # extra stuff in filename
    
    thisTable = opensim.TimeSeriesTable(outFilePath)
    results = {} ;
    results['time'] = np.asarray(thisTable.getIndependentColumn())
    nSteps = len(results['time'])
    temp_r = thisTable.getDependentColumn('walker_knee_r_on_tibia_r_in_tibia_r_mx')
    temp_l = thisTable.getDependentColumn('walker_knee_l_on_tibia_l_in_tibia_l_mx')
    
    results['KAM_r'] = np.ndarray((nSteps))
    results['KAM_l'] = np.ndarray((nSteps))
    for i in range(nSteps):
        results['KAM_r'][i] = -temp_r[i]
        results['KAM_l'][i] = temp_l[i]
    
    return results

# %%
def computeMCF(outputDir,modelPath,activationsPath,IKPath,GRFPath,grfType, 
               muscleForceFilePath = None, pathReserveGeneralizedForces=None, Qds=[],pathJRAResults=None, 
               replaceMuscles = False, visualize=False):
    
    muscForceOverride = muscleForceFilePath!=None
    usingGenForceActuators = pathReserveGeneralizedForces !=None

    if pathJRAResults == None: # If JRA is already computed, skip to MCF computation
        baseRelPath = getBaseDir()
        jointReactionXmlPath = os.path.join(baseRelPath,'opensimPipeline\JointReaction\JointReactionSetup.xml')
        print('you supplied muscle forces for computeMCF - these will override forces computed by OpenSim muscle model.')
        
        
        # %% Editable variables
        statesInDegrees = True # defaults to reading input file, but if not in header, uses this
        removeSpheres = True # Delete spheres as force elements
        # grfType = 'sphere' # Options: 'sphere','sphereResultant','experimental'
        
# %%    
        # Load model
        opensim.Logger.setLevelString('error')
        model = opensim.Model(modelPath)
        
        # Remove spheres                                        
        forceSet = model.getForceSet()
        
        i = 0
        while i < forceSet.getSize():
            if 'SmoothSphere' in forceSet.get(i).getConcreteClassName():
                forceSet.remove(i)
            else:
                i+=1
          
        coords = model.getCoordinateSet()
        nCoords = coords.getSize()
        coordNames = [coords.get(i).getName() for i in range(nCoords)]
        
        forceIndsToRemove = []
        for iF,force in enumerate(forceSet):
            if force.getConcreteClassName() == 'CoordinateActuator':
               forceIndsToRemove.append(iF)
               
        forceIndsToRemove.sort(reverse=True)
        for iF in forceIndsToRemove:
            fName = forceSet.get(iF).getName()
            if pathReserveGeneralizedForces != None: 
                forceSet.remove(iF)
                print('deleted ' + fName + ' from force set. Will be added back if a reserve actuator.')
            else:
                print('did not delete ' + fName + ' from force set but it is not actuated.')
            
        # Add coordinate actuators if they are provided in pathReserveGeneralizedForces
        actuatorNames = []
        if pathReserveGeneralizedForces !=None:
            idTable = opensim.TimeSeriesTable(pathReserveGeneralizedForces) # we just call this ID, because thats what was used in KAM fxn
            # idNames = idTable.getColumnLabels()
            idTime = idTable.getIndependentColumn()
            idLabels = idTable.getColumnLabels()
            
            coordsWithReserves = [[c for c in coordNames if c in f] for f in idLabels]
            coordsWithReserves = [x for x in coordsWithReserves if x != []]
            
            for coordName in coordsWithReserves:
                if isinstance(coordName, list) and len(coordName)==1:
                    coordName = coordName[0]
                newActuator = opensim.CoordinateActuator(coordName)
                newActuator.setName(coordName + '_actuator')
                actuatorNames.append(coordName + '_actuator')
                newActuator.set_min_control(-np.inf)
                newActuator.set_max_control(np.inf)
                newActuator.set_optimal_force(1)
                model.addForce(newActuator)
        
                # Add prescribed controllers for any reserve actuators  
                # Construct constant function
                constFxn = opensim.Constant(0) 
                constFxn.setName(force.getName() + '_constFxn') 
                 
                # Construct prescribed controller
                pController = opensim.PrescribedController() 
                pController.setName(coordName + '_controller') 
                pController.addActuator(newActuator) 
                pController.prescribeControlForActuator(0,constFxn) # attach the function to the controller
                model.addController(pController) 
            
        controllerSet = model.getControllerSet()
        muscles = model.getMuscles()
        
        # Replace muscles
        if replaceMuscles:
            opensim.DeGrooteFregly2016Muscle.replaceMuscles(model,False)
            for force in forceSet:
                if not force.getConcreteClassName() == 'DeGrooteFregly2016Muscle':
                    continue
                c_force = opensim.DeGrooteFregly2016Muscle.safeDownCast(force)
                # Set to default values or values equivalent to what is used in
                # present code.
                c_force.set_passive_fiber_strain_at_one_norm_force(0.6)
                c_force.set_tendon_strain_at_one_norm_force(0.047359470392808856)

        # Load kinematic states, compute speeds
        stateTable = opensim.TimeSeriesTable(IKPath)
        stateNames = stateTable.getColumnLabels()
        stateTime = stateTable.getIndependentColumn()
        
        # Load activation data - if different than IK for some reason
        actTable = opensim.TimeSeriesTable(IKPath)
        # actNames = actTable.getColumnLabels()
        actTime = actTable.getIndependentColumn()        
        
        # Load force data if provided
        if muscForceOverride:
            overrideForceTable = opensim.TimeSeriesTable(muscleForceFilePath)
            overrideForceTime = overrideForceTable.getIndependentColumn()  
        
        try:
            inDegrees = stateTable.getTableMetaDataAsString('inDegrees') == 'yes'
        except:
            inDegrees = statesInDegrees
            print('using supplied statesInDegrees variable, which says statesInDegrees is ' + str(statesInDegrees))
        q = np.zeros((len(stateTime),nCoords))
        dt = stateTime[1] - stateTime[0]    
        if len(Qds) > 0:
            qd_t = np.zeros((len(stateTime),nCoords))
            
        for col in stateNames:
            if 'activation' in col:
                stateTable.removeColumn(col)
            else:
                coordCol = coordNames.index(col)
                for t in range(len(stateTime)):
                    qTemp = np.asarray(stateTable.getDependentColumn(col)[t])
                    
                    if coords.get(col).getMotionType() == 1 and inDegrees: # rotation
                        qTemp = np.deg2rad(qTemp)
                    q[t,coordCol] = copy.deepcopy(qTemp)
                if len(Qds) > 0:
                    # get index
                    idx_col = stateNames.index(col)
                    qd_t[:,coordCol] = Qds[:, idx_col]
        # Add option to pass q_dot as argument.
        if not len(Qds) > 0:
            qd = np.diff(q, axis=0, prepend=np.reshape(q[0,:],(1,nCoords))) / dt
        else:
            qd = qd_t
        # qdd = np.diff(qd, axis=0, prepend=np.reshape(qd[0,:],(1,nCoords))) / dt # just for testing    
        
        # Load activations and muscle forces if overrieing
        muscleNames = []
        act = np.zeros((len(actTime),muscles.getSize()))
        overrideForce = np.copy(act)
        for iMusc,musc in enumerate(muscles):
           muscleNames.append(musc.getName())
           if muscForceOverride:
               forceColName = musc.getName()
               for t in range(len(overrideForceTime)):
                   overrideForce[t,iMusc] = copy.deepcopy(np.asarray(
                                  overrideForceTable.getDependentColumn(forceColName)[t]))
           else:
               actColName = musc.getName()+'/activation'
               for t in range(len(actTime)):
                   act[t,iMusc] = copy.deepcopy(np.asarray(
                                  actTable.getDependentColumn(actColName)[t]))
        
        # Load and apply GRFs
        dataSource = opensim.Storage(GRFPath) 
        
        if grfType == 'sphere':
            appliedToBody = ['calcn','calcn','calcn','calcn','toes','toes']
            for leg in ['r','l']:
                for i in range(len(appliedToBody)):
                    newForce = opensim.ExternalForce()
                    newForce.setName('sphere' + leg + str(i+1)) 
                    newForce.set_applied_to_body(appliedToBody[i] + '_' + leg) 
                    newForce.set_force_expressed_in_body('ground') 
                    newForce.set_point_expressed_in_body('ground')
                    newForce.set_force_identifier('ground_force_s' + str(i+1) + '_' + leg + '_v')
                    newForce.set_torque_identifier('ground_torque_s' + str(i+1) + '_' + leg + '_') ;
                    newForce.set_point_identifier('ground_force_s' + str(i+1) + '_' + leg + '_p') ;
                    newForce.setDataSource(dataSource) ;
                    if removeSpheres:
                        model.addForce(newForce) ;
                    elif i==0:
                        print('GRFs were not applied b/c sphere contacts were used.')
        elif grfType == 'sphereResultant':
            appliedToBody = ['calcn']
            for leg in ['r','l']:
                for i in range(len(appliedToBody)):
                    newForce = opensim.ExternalForce()
                    newForce.setName('GRF' + '_' + leg + '_' + str(i)) 
                    newForce.set_applied_to_body(appliedToBody[i] + '_' + leg.lower()) 
                    newForce.set_force_expressed_in_body('ground') 
                    newForce.set_point_expressed_in_body('ground')
                    newForce.set_force_identifier('ground_force_' + leg + '_v')
                    newForce.set_torque_identifier('ground_torque_' + leg + '_') ;
                    newForce.set_point_identifier('ground_force_' + leg + '_p') ;
                    newForce.setDataSource(dataSource) ;
                    model.addForce(newForce) ; #tested and these make a difference
        elif grfType == 'experimental':
            appliedToBody = ['calcn']
            for leg in ['R','L']:
                for i in range(len(appliedToBody)):
                    newForce = opensim.ExternalForce()
                    newForce.setName('GRF' + '_' + leg + '_' + str(i)) 
                    newForce.set_applied_to_body(appliedToBody[i] + '_' + leg.lower()) 
                    newForce.set_force_expressed_in_body('ground') 
                    newForce.set_point_expressed_in_body('ground')
                    newForce.set_force_identifier(leg + '_ground_force_v')
                    newForce.set_torque_identifier(leg + '_ground_torque_') ;
                    newForce.set_point_identifier(leg + '_ground_force_p') ;
                    newForce.setDataSource(dataSource) ;
                    model.addForce(newForce) ; #tested and these make a difference
                
        
        # initSystem - done editing model
        state = model.initSystem()
        
        # Create state Y map
        yNames = opensim.createStateVariableNamesInSystemOrder(model)
        systemPositionInds = []
        systemVelocityInds = []
        stateNameList = []
        for stateName in coordNames:
            posIdx =np.squeeze(np.argwhere([stateName + '/value' in y for y in yNames]))
            velIdx =np.squeeze(np.argwhere([stateName + '/speed' in y for y in yNames])) 
            if posIdx.size>0:  
                systemPositionInds.append(posIdx)
                systemVelocityInds.append(velIdx)
                stateNameList.append(stateName)
                
        systemActivationInds = []
        for muscleName in muscleNames:
            actIdx = np.squeeze(np.argwhere([muscleName + '/activation' in y for y in yNames]))
            systemActivationInds.append(actIdx)
        
        # Create JRA reporter
        jointReaction = opensim.JointReaction(jointReactionXmlPath)
        model.addAnalysis(jointReaction) ;
        jointReaction.setModel(model) ;
        jointReaction.printToXML(os.path.join(outputDir, 'JrxnSetup.xml')) ;
        
        endTime = []
        if usingGenForceActuators:
            endTime.append(idTime[-1])
        if muscForceOverride:
            endTime.append(overrideForceTable.getIndependentColumn()[-1])
        if len(endTime) == 0:
            endTime = stateTime[-1]
        else:
            endTime = np.max(endTime)
        

        # Loop over time
        if usingGenForceActuators:
            controls = opensim.Vector(nCoords,0) ;
        for iTime in range(len(stateTime)):
            thisTime = stateTime[iTime]
        
            if thisTime <= endTime:
                if usingGenForceActuators:
                    idRow = idTable.getNearestRowIndexForTime(thisTime)  
                if muscForceOverride:
                    overrideForceRow = overrideForceTable.getNearestRowIndexForTime(thisTime)
                actRow = actTable.getNearestRowIndexForTime(thisTime)
                
                # Set time
                state.setTime(thisTime)   
                    
                # Set state, velocity, actuator controls
                yVec = np.zeros((state.getNY())).tolist()
                for iCoord, coord in enumerate(coords):
                    if '_beta' not in coord.getName():
                        
        
                        # Loop thru states to set speeds, vels, activations
                        yVec[systemPositionInds[iCoord]] = q[iTime,iCoord]
                        yVec[systemVelocityInds[iCoord]] = qd[iTime,iCoord]
                                  
                        # Old/slow setting of speeds and velocity
                        # coord.setValue(state,q[iTime,iCoord])
                        # coord.setSpeedValue(state,qd[iTime,iCoord])
                        
                        # if coord.getMotionType() == 1: # rotation
                        #     suffix = '_moment'
                        # elif coord.getMotionType() == 2: # translation
                        #     suffix = '_force'
                        
                        if usingGenForceActuators:
                            actuatorName = [aN for aN in idLabels if coord.getName() in aN] # is there a coord actuator for this coord?
                            if len(actuatorName) == 1:    
                                # Set prescribed controller constant value to control value. Controls
                                # don't live through joint reaction analysis for some reason. 
                                thisController = opensim.PrescribedController.safeDownCast(controllerSet.get(coord.getName() + '_controller')) 
                                thisConstFxn = opensim.Constant.safeDownCast(thisController.get_ControlFunctions(0).get(0))
                                thisConstFxn.setValue(idTable.getDependentColumn(actuatorName[0])[idRow])
                                
                                # Setting controls this way is redundant, but necessary if want to do a force reporter
                                # in future
                                controls.set(iCoord, idTable.getDependentColumn(actuatorName[0])[idRow])
                
                # Set muscle activations or force          
                for iMusc,muscName in enumerate(muscleNames):
                    if muscForceOverride:
                        muscles.get(iMusc).overrideActuation(state,True)
                        muscles.get(iMusc).setOverrideActuation(state,overrideForceTable.getDependentColumn(muscName)[overrideForceRow])
                    else:
                       yVec[systemActivationInds[iMusc]] = act[actRow,iMusc]                         
                
                state.setY(opensim.Vector(yVec))
                               
                model.realizeVelocity(state)
                if not muscForceOverride:
                    model.equilibrateMuscles(state) # this changes muscles to appropriate l and v (not 0 from yVec)
                
                if usingGenForceActuators:
                    model.setControls(state,controls) # tested and commenting didn't impact JRA
                
                # Realize acceleration
                model.realizeAcceleration(state)
                
            # Compute JRA
            if iTime == 0:
                jointReaction.begin(state) 
            else:
                jointReaction.step(state,iTime) 
            if iTime == len(stateTime)-1 or thisTime >= endTime:
                jointReaction.end(state)
        
        # Finish time loop and output
        if not removeSpheres:
            grfType = 'spheresUsed_noGRFsApplied'
        outFileBase = 'results_JRAforMCF'
        jointReaction.printResults(outFileBase,outputDir,-1,'.sto')        
        
        # Get filename
        pathJRAResults = glob.glob(os.path.join(outputDir,outFileBase + '_JointReactionAnalysis_ReactionLoads.sto'))[0] # extra stuff in filename
        
    # Load JRA results values and compute MCF    
    thisTable = opensim.TimeSeriesTable(pathJRAResults)
    results = {} ;
    results['time'] = np.asarray(thisTable.getIndependentColumn())
    nSteps = len(results['time'])
    KAM_r = thisTable.getDependentColumn('walker_knee_r_on_tibia_r_in_tibia_r_mx')
    KAM_l = thisTable.getDependentColumn('walker_knee_l_on_tibia_l_in_tibia_l_mx')
    Fy_r = thisTable.getDependentColumn('walker_knee_r_on_tibia_r_in_tibia_r_fy')
    Fy_l = thisTable.getDependentColumn('walker_knee_l_on_tibia_l_in_tibia_l_fy')
    
    d = 0.04 # intercondyler distance (Lerner 2015)
    
    results['MCF_r'] = np.ndarray((nSteps))
    results['MCF_l'] = np.ndarray((nSteps))
    # can delete these - just for visualization
    temp_Fy = np.ndarray((nSteps))
    temp_KAM = np.ndarray((nSteps))

    # MCF = Fy/2 + KAM/d. Sign changes are due to signs in opensim
    for i in range(nSteps):
        results['MCF_r'][i] = -Fy_r[i]/2 - KAM_r[i]/d
        results['MCF_l'][i] = -Fy_l[i]/2 + KAM_l[i]/d
        temp_Fy[i] = -Fy_l[i]
        temp_KAM[i] = KAM_l[i]/d
        
        
    
    # TODO DELETE
    if visualize:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(temp_Fy)
        plt.plot(results['MCF_l'])
        plt.plot(temp_KAM)
        plt.legend(['fy','mcf','kam/d'])
            
    return results


# TODO delete: Testing
# trial = 'walking2'
# subject = 'subject11'

# basePath = 'C:/MyDriveSym/mobilecap/Data/'+subject+'/OpenSimData/Video/mmpose_0.8/2-cameras/separateLowerUpperBody_OpenPose'
# modelPath = os.path.join(basePath,'Model','LaiArnoldModified2017_poly_withArms_weldHand','LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim')
# outputDir = os.path.join(basePath,'DC','LaiArnoldModified2017_poly_withArms_weldHand',trial+'_videoAndMocap')
# activationsPath = os.path.join(outputDir,'kinematics_act_' + trial +'_videoAndMocap_208.mot')
# IKPath = activationsPath
# IDPath = os.path.join(outputDir,'forces_' + trial + '_videoAndMocap_208.mot')
# GRFPath = os.path.join(outputDir,'GRF_' + trial + '_videoAndMocap_208.mot')

# # pathJRAResults = os.path.join(outputDir,'results_JRA_JointReactionAnalysis_ReactionLoads.sto')
# # pathJRAResults = os.path.join(outputDir,'results_JRAforMCF_JointReactionAnalysis_ReactionLoads.sto')

# forceFilePath = os.path.join(outputDir,'forces_' + trial + '_videoAndMocap_208.mot')
# computeMCF(outputDir,modelPath,activationsPath,IKPath,GRFPath,grfType='sphere', 
#                 muscleForceFilePath = forceFilePath, pathReserveGeneralizedForces=forceFilePath,  
#                 Qds=[],pathJRAResults=None, replaceMuscles = True, visualize=True)
