import casadi as ca
import numpy as np

def polynomialApproximation(musclesPolynomials, polynomialData, NPolynomial):    
    
    from polynomialsOpenSimAD import polynomials
    
    qin = ca.SX.sym('qin', 1, NPolynomial)
    qdotin  = ca.SX.sym('qdotin', 1, NPolynomial)
    lMT = ca.SX(len(musclesPolynomials), 1)
    vMT = ca.SX(len(musclesPolynomials), 1)
    dM = ca.SX(len(musclesPolynomials), NPolynomial)
    
    for count, musclePolynomials in enumerate(musclesPolynomials):
        
        coefficients = polynomialData[musclePolynomials]['coefficients']
        dimension = polynomialData[musclePolynomials]['dimension']
        order = polynomialData[musclePolynomials]['order']        
        spanning = polynomialData[musclePolynomials]['spanning']          
        
        polynomial = polynomials(coefficients, dimension, order)
        
        idxSpanning = [i for i, e in enumerate(spanning) if e == 1]        
        lMT[count] = polynomial.calcValue(qin[0, idxSpanning])
        
        dM[count, :] = 0
        vMT[count] = 0        
        for i in range(len(idxSpanning)):
            dM[count, idxSpanning[i]] = - polynomial.calcDerivative(
                    qin[0, idxSpanning], i)
            vMT[count] += (-dM[count, idxSpanning[i]] * 
               qdotin[0, idxSpanning[i]])
        
    f_polynomial = ca.Function('f_polynomial',[qin, qdotin],[lMT, vMT, dM])
    
    return f_polynomial
        

def hillEquilibrium(mtParameters, tendonCompliance, tendonShift,
                    specificTension, ignorePassiveFiberForce=False):
    
    NMuscles = mtParameters.shape[1]
    # Function variables
    activation = ca.SX.sym('activation', NMuscles)
    mtLength = ca.SX.sym('mtLength', NMuscles)
    mtVelocity = ca.SX.sym('mtVelocity', NMuscles)
    normTendonForce = ca.SX.sym('normTendonForce', NMuscles)
    normTendonForceDT = ca.SX.sym('normTendonForceDT', NMuscles)
     
    hillEquilibrium = ca.SX(NMuscles, 1)
    tendonForce = ca.SX(NMuscles, 1)
    activeFiberForce = ca.SX(NMuscles, 1)
    normActiveFiberLengthForce = ca.SX(NMuscles, 1)
    passiveFiberForce = ca.SX(NMuscles, 1)
    normFiberLength = ca.SX(NMuscles, 1)
    fiberVelocity = ca.SX(NMuscles, 1)
    activeFiberForcePen = ca.SX(NMuscles, 1)
    passiveFiberForcePen = ca.SX(NMuscles, 1)
    
    from muscleModelOpenSimAD import muscleModel
    for m in range(NMuscles):    
        muscle = muscleModel(mtParameters[:, m], activation[m], mtLength[m],
                             mtVelocity[m], normTendonForce[m], 
                             normTendonForceDT[m], tendonCompliance[:, m],
                             tendonShift[:, m], specificTension[:, m],
                             ignorePassiveFiberForce=ignorePassiveFiberForce)
        
        hillEquilibrium[m] = muscle.deriveHillEquilibrium()
        tendonForce[m] = muscle.getTendonForce()
        activeFiberForce[m] = muscle.getActiveFiberForce()[0]
        passiveFiberForce[m] = muscle.getPassiveFiberForce()[0]
        normActiveFiberLengthForce[m] = muscle.getActiveFiberLengthForce()
        normFiberLength[m] = muscle.getFiberLength()[1]
        fiberVelocity[m] = muscle.getFiberVelocity()[0]
        
        activeFiberForcePen[m] = muscle.getActiveFiberForce()[2]
        passiveFiberForcePen[m] = muscle.getPassiveFiberForce()[2]
        
    f_hillEquilibrium = ca.Function('f_hillEquilibrium',
                                    [activation, mtLength, mtVelocity, 
                                     normTendonForce, normTendonForceDT], 
                                     [hillEquilibrium, tendonForce,
                                      activeFiberForce, passiveFiberForce,
                                      normActiveFiberLengthForce,
                                      normFiberLength, fiberVelocity,
                                      activeFiberForcePen,
                                      passiveFiberForcePen]) 
    
    return f_hillEquilibrium

def armActivationDynamics(NArmJoints):
    t = 0.035 # time constant       
    
    eArm = ca.SX.sym('eArm',NArmJoints)
    aArm = ca.SX.sym('aArm',NArmJoints)
    
    aArmDt = (eArm - aArm) / t
    
    f_armActivationDynamics = ca.Function('f_armActivationDynamics',
                                          [eArm, aArm], [aArmDt])
    
    return f_armActivationDynamics  

def metabolicsBhargava(slowTwitchRatio, maximalIsometricForce,
                       muscleMass, smoothingConstant,
                       use_fiber_length_dep_curve=False,
                       use_force_dependent_shortening_prop_constant=True,
                       include_negative_mechanical_work=False):
    
    NMuscles = maximalIsometricForce.shape[0]
    
    # Function variables
    excitation = ca.SX.sym('excitation', NMuscles)
    activation = ca.SX.sym('activation', NMuscles)
    normFiberLength = ca.SX.sym('normFiberLength', NMuscles)
    fiberVelocity = ca.SX.sym('fiberVelocity', NMuscles)
    activeFiberForce = ca.SX.sym('activeFiberForce', NMuscles)
    passiveFiberForce = ca.SX.sym('passiveFiberForce', NMuscles)
    normActiveFiberLengthForce = (
            ca.SX.sym('normActiveFiberLengthForce', NMuscles))
    
    activationHeatRate = ca.SX(NMuscles, 1)
    maintenanceHeatRate = ca.SX(NMuscles, 1)
    shorteningHeatRate = ca.SX(NMuscles, 1)
    mechanicalWork = ca.SX(NMuscles, 1)
    totalHeatRate = ca.SX(NMuscles, 1) 
    metabolicEnergyRate = ca.SX(NMuscles, 1) 
    slowTwitchExcitation = ca.SX(NMuscles, 1) 
    fastTwitchExcitation = ca.SX(NMuscles, 1) 
    
    from metabolicEnergyModel import smoothBhargava2004
    
    for m in range(NMuscles):   
        metabolics = (smoothBhargava2004(excitation[m], activation[m], 
                                         normFiberLength[m],
                                         fiberVelocity[m],
                                         activeFiberForce[m], 
                                         passiveFiberForce[m],
                                         normActiveFiberLengthForce[m],
                                         slowTwitchRatio[m], 
                                         maximalIsometricForce[m],
                                         muscleMass[m], smoothingConstant))
        
        slowTwitchExcitation[m] = metabolics.getTwitchExcitation()[0] 
        fastTwitchExcitation[m] = metabolics.getTwitchExcitation()[1] 
        activationHeatRate[m] = metabolics.getActivationHeatRate()        
        maintenanceHeatRate[m] = metabolics.getMaintenanceHeatRate(
                use_fiber_length_dep_curve)        
        shorteningHeatRate[m] = metabolics.getShorteningHeatRate(
                use_force_dependent_shortening_prop_constant)        
        mechanicalWork[m] = metabolics.getMechanicalWork(
                include_negative_mechanical_work)        
        totalHeatRate[m] = metabolics.getTotalHeatRate()
        metabolicEnergyRate[m] = metabolics.getMetabolicEnergyRate()
        
#    basal_coef = 1.2 # default in OpenSim
#    basal_exp = 1 # default in OpenSim
#    energyModel = (basal_coef * np.power(modelMass, basal_exp) + 
#                   np.sum(metabolicEnergyRate))
    
    f_metabolicsBhargava = ca.Function('metabolicsBhargava',
                                    [excitation, activation, normFiberLength, 
                                     fiberVelocity, activeFiberForce, 
                                     passiveFiberForce, 
                                     normActiveFiberLengthForce], 
                                     [activationHeatRate, maintenanceHeatRate,
                                      shorteningHeatRate, mechanicalWork, 
                                      totalHeatRate, metabolicEnergyRate])
    
    return f_metabolicsBhargava

def limitTorque(k, theta, d):
    
    # Function variables
    Q = ca.SX.sym('Q', 1)
    Qdot = ca.SX.sym('Qdot', 1)
    
    passiveJointTorque = (k[0] * np.exp(k[1] * (Q - theta[1])) + k[2] * 
                           np.exp(k[3] * (Q - theta[0])) - d * Qdot)
    
    f_passiveJointTorque = ca.Function('f_passiveJointTorque', [Q, Qdot], 
                                       [passiveJointTorque])
    
    return f_passiveJointTorque

def passiveTorque(k, d):
    # Function variables
    Q = ca.SX.sym('Q', 1)
    Qdot = ca.SX.sym('Qdot', 1)
    
    passiveJointTorque = -k * Q - d * Qdot
    f_passiveMtpTorque = ca.Function('f_passiveMtpTorque', [Q, Qdot], 
                                     [passiveJointTorque])
    
    return f_passiveMtpTorque   

def normSumPow(N, exp):
    
    # Function variables
    x = ca.SX.sym('x', N,  1)
      
    nsp = ca.sum1(x**exp)       
    nsp = nsp / N
    
    f_normSumPow = ca.Function('f_normSumPow', [x], [nsp])
    
    return f_normSumPow

def normSumWeightedPow(N, exp):
    
    # Function variables
    x = ca.SX.sym('x', N,  1)
    w = ca.SX.sym('w', N,  1)
      
    nsp = ca.sum1(w * (x**exp))       
    nsp = nsp / N
    
    f_normSumPow = ca.Function('f_normSumWeightedPow', [x, w], [nsp])
    
    return f_normSumPow

def normSumPowDev(N, exp, ref):
    
    # Function variables
    x = ca.SX.sym('x', N,  1) 
    
    nsp = ca.sum1((x-ref)**exp)        
    nsp = nsp / N
    
    f_normSumPowDev = ca.Function('f_normSumPowDev', [x], [nsp])
    
    return f_normSumPowDev

def normSumSqr(N):
    
    # Function variables
    x = ca.SX.sym('x', N, 1)
    
    ss = ca.sumsqr(x) / N
        
    f_normSumSqr = ca.Function('f_normSumSqr', [x], [ss])
    
    return f_normSumSqr

def diffTorques():
    
    # Function variables
    jointTorque = ca.SX.sym('x', 1) 
    muscleTorque = ca.SX.sym('x', 1) 
    passiveTorque = ca.SX.sym('x', 1)
    
    diffTorque = jointTorque - (muscleTorque + passiveTorque)
    
    f_diffTorques = ca.Function(
            'f_diffTorques', [jointTorque, muscleTorque, passiveTorque], 
            [diffTorque])
        
    return f_diffTorques

def normSumSqrDiff(dim):
    
    # Function variables
    x = ca.SX.sym('x', dim, 1) 
    x_ref = ca.SX.sym('x_ref', dim, 1)  
    
    nSD = ca.sumsqr(x-x_ref)
    nSD = nSD / dim
        
    f_normSumSqrDiff = ca.Function('f_normSumSqrDiff', [x, x_ref], [nSD])
    
    return f_normSumSqrDiff

def normSumWeightedSqrDiff(dim):
    
    # Function variables
    x = ca.SX.sym('x', dim, 1) 
    x_ref = ca.SX.sym('x_ref', dim, 1)  
    w = ca.SX.sym('w', dim, 1) 
    
    nSD = ca.sum1(w * (x-x_ref)**2)
    nSD = nSD / dim
        
    f_normSumSqrDiff = ca.Function('f_normSumSqrDiff', [x, x_ref, w], [nSD])
    
    return f_normSumSqrDiff

def normSumWeightedPowDiff(power, dim):
    
    # Function variables
    x = ca.SX.sym('x', dim, 1) 
    x_ref = ca.SX.sym('x_ref', dim, 1)  
    w = ca.SX.sym('w', dim, 1) 
    
    nSD = ca.sum1(w * (x-x_ref)**power)
    nSD = nSD / dim
        
    f_normSumSqrDiff = ca.Function('f_normSumSqrDiff', [x, x_ref, w], [nSD])
    
    return f_normSumSqrDiff

def normSumSqrDiffStd(dim):
    
    # Function variables
    x = ca.SX.sym('x', dim, 1) 
    x_ref = ca.SX.sym('x_ref', dim, 1)  
    x_std = ca.SX.sym('x_std', dim, 1)  
    
    nSD = ca.sum1(((x-x_ref)/x_std)**2)
    nSD = nSD / dim
        
    f_normSumSqrDiffStd = ca.Function('f_normSumSqrDiffStd', [x, x_ref, x_std],
                                      [nSD])
    
    return f_normSumSqrDiffStd

def normSumWeightedSqrDiffStd(dim):
    
    # Function variables
    x = ca.SX.sym('x', dim, 1) 
    x_ref = ca.SX.sym('x_ref', dim, 1)  
    x_std = ca.SX.sym('x_std', dim, 1)
    w = ca.SX.sym('w', dim, 1) 
    
    nSD = ca.sum1(w * ((x-x_ref)/x_std)**2)
    nSD = nSD / dim
        
    f_normSumSqrDiffStd = ca.Function('f_normSumSqrDiffStd', 
                                      [x, x_ref, x_std, w], [nSD])
    
    return f_normSumSqrDiffStd

def smoothSphereHalfSpaceForce(transitionVelocity,
                 staticFriction, dynamicFriction, viscousFriction, normal):
    
    dissipation = ca.SX.sym('dissipation', 1) 
    stiffness = ca.SX.sym('stiffness', 1) 
    radius = ca.SX.sym('radius', 1)     
    locSphere_inB = ca.SX.sym('locSphere_inB', 3) 
    posB_inG = ca.SX.sym('posB_inG', 3) 
    lVelB_inG = ca.SX.sym('lVelB_inG', 3) 
    aVelB_inG = ca.SX.sym('aVelB_inG', 3) 
    RBG_inG = ca.SX.sym('RBG_inG', 3, 3) 
    TBG_inG = ca.SX.sym('TBG_inG', 3) 
    
    from contactModel import smoothSphereHalfSpaceForce_ca
    
    contactElement = smoothSphereHalfSpaceForce_ca(stiffness, radius, dissipation,
                                                transitionVelocity,
                                                staticFriction,
                                                dynamicFriction,
                                                viscousFriction, normal)
    
    contactForce = contactElement.getContactForce(locSphere_inB, posB_inG,
                                                  lVelB_inG, aVelB_inG,
                                                  RBG_inG, TBG_inG)
    
    f_smoothSphereHalfSpaceForce = ca.Function(
            'f_smoothSphereHalfSpaceForce',[dissipation, stiffness, radius, locSphere_inB,
                                            posB_inG, lVelB_inG, aVelB_inG,
                                            RBG_inG, TBG_inG], [contactForce])
    
    return f_smoothSphereHalfSpaceForce

def smoothSphereHalfSpaceForce2(normal):
    
    dissipation = ca.SX.sym('dissipation', 1) 
    stiffness = ca.SX.sym('stiffness', 1) 
    radius = ca.SX.sym('radius', 1)     
    locSphere_inB = ca.SX.sym('locSphere_inB', 3) 
    posB_inG = ca.SX.sym('posB_inG', 3) 
    lVelB_inG = ca.SX.sym('lVelB_inG', 3) 
    aVelB_inG = ca.SX.sym('aVelB_inG', 3) 
    RBG_inG = ca.SX.sym('RBG_inG', 3, 3) 
    TBG_inG = ca.SX.sym('TBG_inG', 3) 
    transitionVelocity = ca.SX.sym('transitionVelocity', 1)
    staticFriction = ca.SX.sym('staticFriction', 1)
    dynamicFriction = ca.SX.sym('dynamicFriction', 1)
    viscousFriction = ca.SX.sym('viscousFriction', 1)
    
    from contactModel import smoothSphereHalfSpaceForce_ca
    
    contactElement = smoothSphereHalfSpaceForce_ca(stiffness, radius, dissipation,
                                                transitionVelocity,
                                                staticFriction,
                                                dynamicFriction,
                                                viscousFriction, normal)
    
    contactForce = contactElement.getContactForce(locSphere_inB, posB_inG,
                                                  lVelB_inG, aVelB_inG,
                                                  RBG_inG, TBG_inG)
    
    f_smoothSphereHalfSpaceForce = ca.Function(
            'f_smoothSphereHalfSpaceForce',
            [dissipation, stiffness, staticFriction, dynamicFriction, 
             viscousFriction, transitionVelocity, radius, locSphere_inB,
             posB_inG, lVelB_inG, aVelB_inG, RBG_inG, TBG_inG], [contactForce])
    
    return f_smoothSphereHalfSpaceForce

def muscleMechanicalWorkRate(NMuscles):   
    
    # Function variables
    fiberVelocity = ca.SX.sym('fiberVelocity', NMuscles)
    activeFiberForce = ca.SX.sym('activeFiberForce', NMuscles)
           
    mechanicalWorkRate = -activeFiberForce * fiberVelocity
        
    f_muscleMechanicalWorkRate = ca.Function('f_muscleMechanicalWorkRate',
                                             [activeFiberForce, fiberVelocity],
                                             [mechanicalWorkRate])
    
    return f_muscleMechanicalWorkRate   

def jointMechanicalWorkRate(NJoints):    
    
    # Function variables
    jointVelocity = ca.SX.sym('jointVelocity', NJoints)
    jointTorque = ca.SX.sym('jointTorque', NJoints)
          
    mechanicalWorkRate = jointTorque * jointVelocity
        
    f_jointMechanicalWorkRate = ca.Function('f_jointMechanicalWorkRate',
                                             [jointTorque, jointVelocity],
                                             [mechanicalWorkRate])
    
    return f_jointMechanicalWorkRate  

# Test f_hillEquilibrium
#import numpy as np
#mtParametersT = np.array([[819, 573, 653],
#                 [0.0520776466291754, 0.0823999283675263, 0.0632190293747345],
#                 [0.0759262885434707, 0.0516827953074425, 0.0518670055241629],
#                 [0.139626340000000, 0,	0.331612560000000],
#                 [0.520776466291754,	0.823999283675263,	0.632190293747345]])
#tendonComplianceT = np.array([35, 35, 35])
#tendonShiftT = np.array([0, 0, 0])
#specificTensionT = np.array([0.74455, 0.75395, 0.75057])
#f_hillEquilibrium = hillEquilibrium(mtParametersT, tendonComplianceT,
#                                    tendonShiftT, specificTensionT)
#
#activationT = [0.8, 0.7, 0.6]
#mtLengthT = [1.2, 0.9, 1.3]
#mtVelocity = [0.8, 0.1, 5.4]
#normTendonForce = [0.8, 0.4, 0.9]
#normTendonForceDT = [2.1, 3.4, -5.6]
#
#hillEquilibriumT = f_hillEquilibrium(activationT, mtLengthT, mtVelocity,
#                                     normTendonForce, normTendonForceDT)[0]
#tendonForceT = f_hillEquilibrium(activationT, mtLengthT, mtVelocity,
#                                 normTendonForce, normTendonForceDT)[1]
#activeFiberForceT = f_hillEquilibrium(activationT, mtLengthT, mtVelocity,
#                                      normTendonForce, normTendonForceDT)[2]
#passiveFiberForceT = f_hillEquilibrium(activationT, mtLengthT, mtVelocity,
#                                     normTendonForce, normTendonForceDT)[3]
#normActiveFiberLengthForceT = f_hillEquilibrium(activationT, mtLengthT,
#                                                mtVelocity, normTendonForce,
#                                                normTendonForceDT)[4]
#maximalFiberVelocityT = f_hillEquilibrium(activationT, mtLengthT, mtVelocity,
#                                          normTendonForce, 
#                                          normTendonForceDT)[5]
#muscleMassT = f_hillEquilibrium(activationT, mtLengthT, mtVelocity, 
#                                normTendonForce, normTendonForceDT)[6]
#normFiberLengthT = f_hillEquilibrium(activationT, mtLengthT, mtVelocity, 
#                                     normTendonForce, normTendonForceDT)[7]
#fiberVelocityT = f_hillEquilibrium(activationT, mtLengthT, mtVelocity, 
#                                   normTendonForce, normTendonForceDT)[8]
#
#print(hillEquilibriumT)
#print(tendonForceT)
#print(activeFiberForceT)
#print(passiveFiberForceT)
#print(normActiveFiberLengthForceT)
#print(maximalFiberVelocityT)
#print(muscleMassT)
#print(normFiberLengthT)
#print(fiberVelocityT)

# Test f_armActivationDynamics
#f_armActivationDynamics = armActivationDynamics(3)
#eArmT = [0.8, 0.6, 0.4]
#aArmT = [0.5, 0.4, 0.3]
#aArmDtT = f_armActivationDynamics(eArmT, aArmT)
#
#print(aArmDtT)
    
#import polynomialData
#from polynomials import polynomials
#polynomialData = polynomialData.polynomialData()
#coefficients = polynomialData['glut_med1_r']['coefficients']
#dimension = polynomialData['glut_med1_r']['dimension']
#order = polynomialData['glut_med1_r']['order']        
#spanning = polynomialData['glut_med1_r']['spanning']   
#
#
#polynomial = polynomials(coefficients, dimension, order)
#
#qin     = ca.SX.sym('qin', 1, len(spanning));
##qdotin  = ca.SX.sym('qdotin', 1, len(spanning));
#
#idxSpanning = [i for i, e in enumerate(spanning) if e == 1]
#
#lmT = polynomial.calcValue(qin[0, idxSpanning])
#
#f_polynomial = ca.Function('f_polynomial',[qin],[lmT])
#
#qinT = [0.814483478343008, 1.05503342897057, 0.162384573599574,
#        0.0633034484654646, 0.433004984392647, 0.716775413397760,
#        -0.0299471169706956, 0.200356847296188, 0.716775413397760]
#lmTT = f_polynomial(qinT)
#print(lmTT)
    
#Qin = 5
#Qdotin = 7
#
#passiveJointTorque_hfT = f_passiveJointTorque_hip_flexion(Qin, Qdotin)
#print(passiveJointTorque_hfT)
#passiveJointTorque_haT = f_passiveJointTorque_hip_adduction(Qin, Qdotin)
#print(passiveJointTorque_haT)
#passiveJointTorque_hrT = f_passiveJointTorque_hip_rotation(Qin, Qdotin)
#print(passiveJointTorque_hrT)
#passiveJointTorque_kaT = f_passiveJointTorque_knee_angle(Qin, Qdotin)
#print(passiveJointTorque_kaT)
#passiveJointTorque_aaT = f_passiveJointTorque_ankle_angle(Qin, Qdotin)
#print(passiveJointTorque_aaT)
#passiveJointTorque_saT = f_passiveJointTorque_subtalar_angle(Qin, Qdotin)
#print(passiveJointTorque_saT)
#passiveJointTorque_leT = f_passiveJointTorque_lumbar_extension(Qin, Qdotin)
#print(passiveJointTorque_leT)
#passiveJointTorque_lbT = f_passiveJointTorque_lumbar_bending(Qin, Qdotin)
#print(passiveJointTorque_lbT)
#passiveJointTorque_lrT = f_passiveJointTorque_lumbar_rotation(Qin, Qdotin)
#print(passiveJointTorque_lrT)