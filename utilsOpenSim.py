import os
import opensim
    
# %% Inverse kinematics.
def runIKTool(pathGenericSetupFile, pathScaledModel, pathTRCFile,
              pathOutputFolder, timeRange=[], IKFileName='not_specified'):
    
    # Paths
    if IKFileName == 'not_specified':
        _, IKFileName = os.path.split(pathTRCFile)
        IKFileName = IKFileName[:-4]
    pathOutputMotion = os.path.join(pathOutputFolder, IKFileName + '.mot')
    pathOutputSetup =  os.path.join(
        pathOutputFolder, 'Setup_InverseKinematics_' + IKFileName + '.xml')
    
    # Setup IK tool.
    opensim.Logger.setLevelString('error')
    IKTool = opensim.InverseKinematicsTool(pathGenericSetupFile)            
    IKTool.setName(IKFileName)
    IKTool.set_model_file(pathScaledModel)          
    IKTool.set_marker_file(pathTRCFile)
    if timeRange:
        IKTool.set_time_range(0, timeRange[0])
        IKTool.set_time_range(1, timeRange[-1])
    IKTool.setResultsDir(pathOutputFolder)                        
    IKTool.set_report_errors(True)
    IKTool.set_report_marker_locations(False)
    IKTool.set_output_motion_file(pathOutputMotion)
    IKTool.printToXML(pathOutputSetup)
    command = 'opensim-cmd -o error' + ' run-tool ' + pathOutputSetup
    os.system(command)
    
    return 