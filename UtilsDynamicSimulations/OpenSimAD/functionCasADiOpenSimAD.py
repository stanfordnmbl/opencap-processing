'''
    ---------------------------------------------------------------------------
    OpenCap processing: functionCasADiOpenSimAD.py
    ---------------------------------------------------------------------------
    Copyright 2022 Stanford University and the Authors
    
    Author(s): Antoine Falisse, Scott Uhlrich
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    
    This script defines several CasADi functions for use when setting up
    the optimal control problem.
'''

import casadi as ca
import numpy as np

# %% CasADi function to approximate muscle-tendon lenghts, velocities,
# and moment arms based on polynomial approximations of joint positions and
# velocities.
def polynomialApproximation(musclesPolynomials, polynomialData, NPolynomial):    
    
    from polynomialsOpenSimAD import polynomials
    
    # Function variables.
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
        
# %% CasADi function to describe the Hill equilibrium based on the
# DeGrooteFregly2016MuscleModel muscle model.
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
    
    from muscleModelOpenSimAD import DeGrooteFregly2016MuscleModel
    for m in range(NMuscles):    
        muscle = DeGrooteFregly2016MuscleModel(
            mtParameters[:, m], activation[m], mtLength[m],
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
    f_hillEquilibrium = ca.Function(
        'f_hillEquilibrium', [activation, mtLength, mtVelocity, 
                              normTendonForce, normTendonForceDT], 
        [hillEquilibrium, tendonForce, activeFiberForce, passiveFiberForce,
         normActiveFiberLengthForce, normFiberLength, fiberVelocity,
         activeFiberForcePen, passiveFiberForcePen]) 
    
    return f_hillEquilibrium

# %% CasADi function to describe the dynamics of the coordinate actuators.
def coordinateActuatorDynamics(nJoints):
    
    # Function variables
    eArm = ca.SX.sym('eArm',nJoints)
    aArm = ca.SX.sym('aArm',nJoints)
    
    t = 0.035 # time constant  
    aArmDt = (eArm - aArm) / t    
    f_armActivationDynamics = ca.Function('f_armActivationDynamics',
                                          [eArm, aArm], [aArmDt])
    
    return f_armActivationDynamics

# %% CasADi function to compute passive limit joint torques.
def limitPassiveTorque(k, theta, d):
    
    # Function variables
    Q = ca.SX.sym('Q', 1)
    Qdot = ca.SX.sym('Qdot', 1)
    
    passiveJointTorque = (k[0] * np.exp(k[1] * (Q - theta[1])) + k[2] * 
                           np.exp(k[3] * (Q - theta[0])) - d * Qdot)    
    f_limitPassiveTorque = ca.Function('f_limitPassiveTorque', [Q, Qdot], 
                                       [passiveJointTorque])
    
    return f_limitPassiveTorque

# %% CasADi function to compute linear passive joint torques given stiffness
# and damping.
def linarPassiveTorque(k, d):
    
    # Function variables
    Q = ca.SX.sym('Q', 1)
    Qdot = ca.SX.sym('Qdot', 1)
    
    passiveJointTorque = -k * Q - d * Qdot
    f_linarPassiveTorque = ca.Function('f_linarPassiveTorque', [Q, Qdot], 
                                       [passiveJointTorque])
    
    return f_linarPassiveTorque

# %% CasADi function to compute the normalized sum of the weighted elements in
# a vector elevated to a given power.
def normSumWeightedPow(N, exp):
    
    # Function variables
    x = ca.SX.sym('x', N,  1)
    w = ca.SX.sym('w', N,  1)
      
    nsp = ca.sum1(w * (x**exp))       
    nsp = nsp / N    
    f_normSumPow = ca.Function('f_normSumWeightedPow', [x, w], [nsp])
    
    return f_normSumPow

# %% CasADi function to compute the normalized sum of the squared elements in a
# vector.
def normSumSqr(N):
    
    # Function variables
    x = ca.SX.sym('x', N, 1)
    
    ss = ca.sumsqr(x) / N        
    f_normSumSqr = ca.Function('f_normSumSqr', [x], [ss])
    
    return f_normSumSqr

# %% CasADi function to compute difference in torques (inverse dynamics vs
# muscle and passive contributions).
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

# %% CasADi function to compute the normalized sum of the weighted squared
# difference between two vectors.
def normSumWeightedSqrDiff(dim):
    
    # Function variables
    x = ca.SX.sym('x', dim, 1) 
    x_ref = ca.SX.sym('x_ref', dim, 1)  
    w = ca.SX.sym('w', dim, 1) 
    
    nSD = ca.sum1(w * (x-x_ref)**2)
    nSD = nSD / dim        
    f_normSumSqrDiff = ca.Function('f_normSumSqrDiff', [x, x_ref, w], [nSD])
    
    return f_normSumSqrDiff