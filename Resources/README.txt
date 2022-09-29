Thank you for using OpenCap!

###################################################################################################
OpenCap is a web-based software package to estimate the kinematics and dynamics of human movement from smartphone videos.
For more information about OpenCap, please visit https://www.opencap.ai/.
You will also find details about the software in our pre-print: https://www.biorxiv.org/content/10.1101/2022.07.07.499061v1. 
Please cite this preprint if you use OpenCap in your research.

###################################################################################################
For technical questions, please use our forum: https://simtk.org/plugins/phpBB/indexPhpbb.php?group_id=2385&pluginname=phpBB.
For other questions, please use our web form: https://www.opencap.ai/#contact.

###################################################################################################
This folder contains data processed with OpenCap.
Here is a brief overview of the folder structure:
- CalibrationImages 
    -> This folder contains images of the checkerboard used for camera calibration.
- MarkerData 
    -> This folder contains marker data; each marker data file is named <trial name>.trc.
        -> Markers with names ending with _study are anatomical markers; other markers are video keypoints. Anatomical markers are predicted from video keypoints (see pre-print for details: https://www.biorxiv.org/content/10.1101/2022.07.07.499061v1).
-  OpenSimData: 
    -> This folder contains data processed with OpenSim.
        -> For more information about OpenSim, please visit https://simtk-confluence.stanford.edu:8443/display/OpenSim.
    - Model
        -> The .osim file is the scaled musculoskeletal model.
            -> The model can be opened in the OpenSim GUI for visualization.
            -> The model can be opened as as text file with any text editor (e.g., Notepad++).
    - Kinematics
        -> This folder contains motion files with the kinematic data; each motion file is named <trial name>.mot.
            -> The motion files can be loaded, together with the scaled musculoskeletal model, in the OpenSim GUI for visualization.
            -> The motion files can be opened as spreadsheets with any spreadsheet software (e.g., Excel).
- Videos
    -> This folder contains raw and processed video data.
        -> There are as many subfolders as there are cameras used for data collection; the subfolders are named Cam<i> where i is the index of the camera.
    - Cam<i>:
        - InputMedia
            -> This folder contains one subfolder per trial; the subfolders are named as per trial name.
            - <trial name>
                -> <trial name>.mov is the raw recorded video.
                -> <trial name>_sync.mp4 is the processed video synced across cameras.
        - OutputPkl
            -> This folder contains Python pickle files with outputs from the pose detection algorithm; each file is named <trial name>_keypoints.pkl.
        - cameraIntrinsicsExtrinsics.pickle
            -> This Python pickle file contains the camera intrinsic and extrinsic parameters.
    - mappingCamDevice.pickle 
        -> This Python pickle file is used internally to map the recorded videos to the recording cameras.
- sessionMetadata.yaml
    -> This file contains metadata about the session.
###################################################################################################
