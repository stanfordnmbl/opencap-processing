#ifndef OPENSIM_MUSCLE_H_
#define OPENSIM_MUSCLE_H_
/* -------------------------------------------------------------------------- *
 *                             OpenSim:  Muscle.h                             *
 * -------------------------------------------------------------------------- *
 * The OpenSim API is a toolkit for musculoskeletal modeling and simulation.  *
 * See http://opensim.stanford.edu and the NOTICE file for more information.  *
 * OpenSim is developed at Stanford University and supported by the US        *
 * National Institutes of Health (U54 GM072970, R24 HD065690) and by DARPA    *
 * through the Warrior Web program.                                           *
 *                                                                            *
 * Copyright (c) 2005-2017 Stanford University and the Authors                *
 * Author(s): Ajay Seth, Matthew Millard                                      *
 *                                                                            *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may    *
 * not use this file except in compliance with the License. You may obtain a  *
 * copy of the License at http://www.apache.org/licenses/LICENSE-2.0.         *
 *                                                                            *
 * Unless required by applicable law or agreed to in writing, software        *
 * distributed under the License is distributed on an "AS IS" BASIS,          *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
 * See the License for the specific language governing permissions and        *
 * limitations under the License.                                             *
 * -------------------------------------------------------------------------- */


// INCLUDE
#include "PathActuator.h"

#ifdef SWIG
    #ifdef OSIMSIMULATION_API
        #undef OSIMSIMULATION_API
        #define OSIMSIMULATION_API
    #endif
#endif

namespace OpenSim {

//==============================================================================
//                              Muscle Exceptions
//==============================================================================
class MuscleCannotEquilibrate : public Exception {
public:
    MuscleCannotEquilibrate(const std::string& file,
                            size_t line,
                            const std::string& func,
                            const Object& obj,
                            const std::string& detail) :
        Exception(file, line, func, obj, detail) {
        std::string msg = "Unable to compute equilibrium for this muscle.\n";
        msg += "Please verify that the initial activation is valid and that ";
        msg += "the length of the musculotendon actuator doesn't produce a ";
        msg += "pennation angle of 90 degrees or a negative fiber length.";
        addMessage(msg);
    }
};

//==============================================================================
//                                    Muscle
//==============================================================================
/**
 * A base class for modeling a muscle-tendon actuator. It defines muscle parameters
 * and methods to PathActuator, but does not implement all of the necessary methods,
 * and remains an abstract class. The path information for a muscle is contained
 * in PathActuator, and the force-generating behavior should be defined in
 * the derived classes.
 *
 * This class defines a subset of muscle models that include an active fiber
 * (contractile element) in series with a tendon. This class defines common 
 * data members and handles the geometry of a unipennate fiber in connection
 * with a tendon. No states are assumed, but concrete classes are free to
 * add whatever states are necessary to describe the specific behavior of a
 * muscle.
 *
 * @author Ajay Seth, Matt Millard
 *
 * (Based on earlier work by Peter Loan and Frank C. Anderson.)
 */
class OSIMSIMULATION_API Muscle : public PathActuator {
OpenSim_DECLARE_ABSTRACT_OBJECT(Muscle, PathActuator);
public:
//=============================================================================
// PROPERTIES
//=============================================================================
    OpenSim_DECLARE_PROPERTY(max_isometric_force, osim_double_adouble,
        "Maximum isometric force that the fibers can generate");
    OpenSim_DECLARE_PROPERTY(optimal_fiber_length, osim_double_adouble,
        "Optimal length of the muscle fibers");
    OpenSim_DECLARE_PROPERTY(tendon_slack_length, osim_double_adouble,
        "Resting length of the tendon");
    OpenSim_DECLARE_PROPERTY(pennation_angle_at_optimal, osim_double_adouble,
        "Angle between tendon and fibers at optimal fiber length expressed in radians");
    OpenSim_DECLARE_PROPERTY(max_contraction_velocity, osim_double_adouble,
        "Maximum contraction velocity of the fibers, in optimal fiberlengths/second");
    OpenSim_DECLARE_PROPERTY(ignore_tendon_compliance, bool,
        "Compute muscle dynamics ignoring tendon compliance. Tendon is assumed to be rigid.");
    OpenSim_DECLARE_PROPERTY(ignore_activation_dynamics, bool,
        "Compute muscle dynamics ignoring activation dynamics. Activation is equivalent to excitation.");

//=============================================================================
// OUTPUTS
//=============================================================================
    //OpenSim_DECLARE_OUTPUT(excitation, osim_double_adouble, getExcitation,
    //        SimTK::Stage::Dynamics);
    OpenSim_DECLARE_OUTPUT(activation, osim_double_adouble, getActivation,
            SimTK::Stage::Dynamics);
    OpenSim_DECLARE_OUTPUT(fiber_length, osim_double_adouble, getFiberLength,
            SimTK::Stage::Position);
    OpenSim_DECLARE_OUTPUT(pennation_angle, osim_double_adouble, getPennationAngle,
            SimTK::Stage::Position);
    OpenSim_DECLARE_OUTPUT(cos_pennation_angle, osim_double_adouble, getCosPennationAngle,
            SimTK::Stage::Position);
    OpenSim_DECLARE_OUTPUT(tendon_length, osim_double_adouble, getTendonLength,
            SimTK::Stage::Position);
    OpenSim_DECLARE_OUTPUT(normalized_fiber_length, osim_double_adouble,
            getNormalizedFiberLength, SimTK::Stage::Position);
    OpenSim_DECLARE_OUTPUT(fiber_length_along_tendon, osim_double_adouble,
            getFiberLengthAlongTendon, SimTK::Stage::Position);
    OpenSim_DECLARE_OUTPUT(tendon_strain, osim_double_adouble, getTendonStrain,
            SimTK::Stage::Position);
    OpenSim_DECLARE_OUTPUT(passive_force_multiplier, osim_double_adouble,
            getPassiveForceMultiplier, SimTK::Stage::Position);
    OpenSim_DECLARE_OUTPUT(active_force_length_multiplier, osim_double_adouble,
            getActiveForceLengthMultiplier, SimTK::Stage::Position);
    OpenSim_DECLARE_OUTPUT(fiber_velocity, osim_double_adouble, getFiberVelocity,
            SimTK::Stage::Velocity);
    OpenSim_DECLARE_OUTPUT(normalized_fiber_velocity, osim_double_adouble,
            getNormalizedFiberVelocity, SimTK::Stage::Velocity);
    OpenSim_DECLARE_OUTPUT(fiber_velocity_along_tendon, osim_double_adouble,
            getFiberVelocityAlongTendon, SimTK::Stage::Velocity);
    OpenSim_DECLARE_OUTPUT(tendon_velocity, osim_double_adouble, getTendonVelocity,
            SimTK::Stage::Velocity);
    OpenSim_DECLARE_OUTPUT(force_velocity_multiplier, osim_double_adouble,
            getForceVelocityMultiplier, SimTK::Stage::Velocity);
    OpenSim_DECLARE_OUTPUT(pennation_angular_velocity, osim_double_adouble,
            getPennationAngularVelocity, SimTK::Stage::Velocity);
    OpenSim_DECLARE_OUTPUT(fiber_force, osim_double_adouble, getFiberForce,
            SimTK::Stage::Dynamics);
    OpenSim_DECLARE_OUTPUT(fiber_force_along_tendon, osim_double_adouble,
            getFiberForceAlongTendon, SimTK::Stage::Dynamics);
    OpenSim_DECLARE_OUTPUT(active_fiber_force, osim_double_adouble, getActiveFiberForce,
            SimTK::Stage::Dynamics);
    OpenSim_DECLARE_OUTPUT(passive_fiber_force, osim_double_adouble, getPassiveFiberForce,
            SimTK::Stage::Dynamics);
    OpenSim_DECLARE_OUTPUT(active_fiber_force_along_tendon, osim_double_adouble,
            getActiveFiberForceAlongTendon, SimTK::Stage::Dynamics);
    OpenSim_DECLARE_OUTPUT(passive_fiber_force_along_tendon, osim_double_adouble,
            getPassiveFiberForceAlongTendon, SimTK::Stage::Dynamics);
    OpenSim_DECLARE_OUTPUT(tendon_force, osim_double_adouble, getTendonForce,
            SimTK::Stage::Dynamics);
    OpenSim_DECLARE_OUTPUT(fiber_stiffness, osim_double_adouble, getFiberStiffness,
            SimTK::Stage::Dynamics);
    OpenSim_DECLARE_OUTPUT(fiber_stiffness_along_tendon, osim_double_adouble,
            getFiberStiffnessAlongTendon, SimTK::Stage::Dynamics);
    OpenSim_DECLARE_OUTPUT(tendon_stiffness, osim_double_adouble, getTendonStiffness,
            SimTK::Stage::Dynamics);
    OpenSim_DECLARE_OUTPUT(muscle_stiffness, osim_double_adouble, getMuscleStiffness,
            SimTK::Stage::Dynamics);
    OpenSim_DECLARE_OUTPUT(fiber_active_power, osim_double_adouble, getFiberActivePower,
            SimTK::Stage::Dynamics);
    OpenSim_DECLARE_OUTPUT(fiber_passive_power, osim_double_adouble, getFiberPassivePower,
            SimTK::Stage::Dynamics);
    OpenSim_DECLARE_OUTPUT(tendon_power, osim_double_adouble, getTendonPower,
            SimTK::Stage::Dynamics);
    OpenSim_DECLARE_OUTPUT(muscle_power, osim_double_adouble, getMusclePower,
            SimTK::Stage::Dynamics);

//=============================================================================
// PUBLIC METHODS
//=============================================================================
    /** @name Constructors and Destructor
     */ 
    //@{
    /** Default constructor. */
    Muscle();

    // default destructor, copy constructor and copy assignment
    
    //--------------------------------------------------------------------------
    // MUSCLE PARAMETER ACCESSORS
    //--------------------------------------------------------------------------
    /** @name Muscle Parameters Access Methods
     */ 
    //@{
    /** get/set the maximum isometric force (in N) that the fibers can generate */
    osim_double_adouble getMaxIsometricForce() const; 
    void setMaxIsometricForce(osim_double_adouble maxIsometricForce);

    /** get/set the optimal length (in m) of the muscle fibers (lumped as a single fiber) */
    osim_double_adouble getOptimalFiberLength() const;
    void setOptimalFiberLength(osim_double_adouble optimalFiberLength);

    /** get/set the resting (slack) length (in m) of the tendon that is in series with the muscle fiber */
    osim_double_adouble getTendonSlackLength() const;
    void setTendonSlackLength(osim_double_adouble tendonSlackLength);

    /** get/set the angle (in radians) between fibers at their optimal fiber length and the tendon */
    osim_double_adouble getPennationAngleAtOptimalFiberLength() const;
    void setPennationAngleAtOptimalFiberLength(osim_double_adouble pennationAngle);
    
    /** get/set the maximum contraction velocity of the fibers, in optimal fiber-lengths per second */
    osim_double_adouble getMaxContractionVelocity() const;
    void setMaxContractionVelocity(osim_double_adouble maxContractionVelocity);

    // End of Muscle Parameter Accessors.
    //@} 

    //--------------------------------------------------------------------------
    // MUSCLE STATE DEPENDENT ACCESSORS
    //--------------------------------------------------------------------------
    /** @name Muscle State Dependent Access Methods
     *  Get quantities of interest common to all muscles
     */ 
    //@{

    /** Get/set Modeling (runtime) option to ignore tendon compliance when 
    computing muscle dynamics. This does not directly modify the persistent
    property value. **/
    bool getIgnoreTendonCompliance(const SimTK::State& s) const;
    void setIgnoreTendonCompliance(SimTK::State& s, bool ignore) const;

    /** Get/set Modeling (runtime) option to ignore activation dynamics when 
    computing muscle dynamics. This does not directly modify the persistent
    property value. **/
    bool getIgnoreActivationDynamics(const SimTK::State& s) const;
    void setIgnoreActivationDynamics(SimTK::State& s, bool ignore) const;

    /** get the activation level of the muscle, which modulates the active force
        of the muscle and has a normalized (0 to 1) value 
        Note: method remains virtual to permit override by deprecated muscles. */
    virtual osim_double_adouble getActivation(const SimTK::State& s) const;

    /** get the current working fiber length (m) for the muscle */
    osim_double_adouble getFiberLength(const SimTK::State& s) const;
    /** get the current pennation angle (radians) between the fiber and tendon at the current fiber length  */
    osim_double_adouble getPennationAngle(const SimTK::State& s) const;
    /** get the cosine of the current pennation angle (radians) between the fiber and tendon at the current fiber length  */
    osim_double_adouble getCosPennationAngle(const SimTK::State& s) const;
    /** get the current tendon length (m)  given the current joint angles and fiber length */
    osim_double_adouble getTendonLength(const SimTK::State& s) const;
    /** get the current normalized fiber length (fiber_length/optimal_fiber_length) */
    osim_double_adouble getNormalizedFiberLength(const SimTK::State& s) const;
    /** get the current fiber length (m) projected (*cos(pennationAngle)) onto the tendon direction */
    osim_double_adouble getFiberLengthAlongTendon(const SimTK::State& s) const;
    /** get the current tendon strain (delta_l/tendon_slack_length is dimensionless)  */
    osim_double_adouble getTendonStrain(const SimTK::State& s) const;

    /** the potential energy (J) stored in the fiber due to its parallel elastic element */
    osim_double_adouble getFiberPotentialEnergy(const SimTK::State& s) const;
    /** the potential energy (J) stored in the tendon */    
    osim_double_adouble getTendonPotentialEnergy(const SimTK::State& s) const;
    /** the total potential energy (J) stored in the muscle */  
    osim_double_adouble getMusclePotentialEnergy(const SimTK::State& s) const;
    
    /** get the passive fiber (parallel elastic element) force multiplier */
    osim_double_adouble getPassiveForceMultiplier(const SimTK::State& s) const;
    /** get the active fiber (contractile element) force multiplier due to current fiber length */
    osim_double_adouble getActiveForceLengthMultiplier(const SimTK::State& s) const;

    /** get current fiber velocity (m/s) positive is lengthening */
    osim_double_adouble getFiberVelocity(const SimTK::State& s) const;
    /** get normalize fiber velocity (fiber_lengths/s / max_contraction_velocity) */
    osim_double_adouble getNormalizedFiberVelocity(const SimTK::State& s) const;
    /** get the current fiber velocity (m/s) projected onto the tendon direction */
    osim_double_adouble getFiberVelocityAlongTendon(const SimTK::State& s) const;
    /** get pennation angular velocity (radians/s) */
    osim_double_adouble getPennationAngularVelocity(const SimTK::State& s) const;
    /** get the tendon velocity (m/s) positive is lengthening */
    osim_double_adouble getTendonVelocity(const SimTK::State& s) const;
    /** get the dimensionless multiplier resulting from the fiber's force-velocity curve */
    osim_double_adouble getForceVelocityMultiplier(const SimTK::State& s) const;

    /** get the current fiber force (N) applied to the tendon */
    osim_double_adouble getFiberForce(const SimTK::State& s) const;
    /**get the force of the fiber (N/m) along the direction of the tendon*/
    osim_double_adouble getFiberForceAlongTendon(const SimTK::State& s) const;
    /** get the current active fiber force (N) due to activation*force_length*force_velocity relationships */
    osim_double_adouble getActiveFiberForce(const SimTK::State& s) const;
    /** get the total force applied by all passive elements in the fiber (N) */
    osim_double_adouble getPassiveFiberForce(const SimTK::State& s) const;
    /** get the current active fiber force (N) projected onto the tendon direction */
    osim_double_adouble getActiveFiberForceAlongTendon(const SimTK::State& s) const;
    /** get the total force applied by all passive elements in the fiber (N)
        projected onto the tendon direction */
    osim_double_adouble getPassiveFiberForceAlongTendon(const SimTK::State& s) const;
    /** get the current tendon force (N) applied to bones */
    osim_double_adouble getTendonForce(const SimTK::State& s) const;

    /** get the current fiber stiffness (N/m) defined as the partial derivative
        of fiber force with respect to fiber length */
    osim_double_adouble getFiberStiffness(const SimTK::State& s) const;
    /**get the stiffness of the fiber (N/m) along the direction of the tendon,
    that is the partial derivative of the fiber force along the tendon with
    respect to small changes in fiber length along the tendon*/
    osim_double_adouble getFiberStiffnessAlongTendon(const SimTK::State& s) const;
    /** get the current tendon stiffness (N/m) defined as the partial derivative
        of tendon force with respect to tendon length */
    osim_double_adouble getTendonStiffness(const SimTK::State& s) const;
    /** get the current muscle stiffness (N/m) defined as the partial derivative
        of muscle force with respect to muscle length */
    osim_double_adouble getMuscleStiffness(const SimTK::State& s) const;

    /** get the current active fiber power (W) */
    osim_double_adouble getFiberActivePower(const SimTK::State& s) const;
    /** get the current passive fiber power (W) */
    osim_double_adouble getFiberPassivePower(const SimTK::State& s) const;

    /** get the current tendon power (W) */
    osim_double_adouble getTendonPower(const SimTK::State& s) const;
    /** get the current muscle power (W) */
    osim_double_adouble getMusclePower(const SimTK::State& s) const;
    
    /** get the stress in the muscle (part of the Actuator interface as well) */
    osim_double_adouble getStress(const SimTK::State& s) const override;
    
    /** set the excitation (control) for this muscle. NOTE if controllers are connected to the
        muscle and are adding in their controls, and setExcitation is called after the model's
        computeControls(), then setExcitation will override the controller values. If called 
        before computeControls, then controller value(s) are added to the excitation set here. */
    void setExcitation(SimTK::State& s, osim_double_adouble excitation) const;
    //osim_double_adouble getExcitation(const SimTK::State& s) const;


    /** DEPRECATED: only for backward compatibility */
    virtual void setActivation(SimTK::State& s, osim_double_adouble activation) const = 0;

    // End of Muscle's State Dependent Accessors.
    //@} 

    /** Actuator interface for a muscle computes the tension in the muscle
        and applied by the tendon to bones (i.e. not the fiber force) */
    //osim_double_adouble computeActuation(const SimTK::State& s) const override = 0;


    /** @name Muscle initialization 
     */ 
    //@{
    /** Find and set the equilibrium state of the muscle (if any) */
    void computeEquilibrium(SimTK::State& s) const override final {
        return computeInitialFiberEquilibrium(s);
    }
    // End of Muscle's State Dependent Accessors.
    //@} 

    ///@cond
    //--------------------------------------------------------------------------
    // Estimate the muscle force for a given activation based on a rigid tendon 
    // assumption and neglecting passive fiber force. This provides a linear 
    // relationship between activation and force. This is used by CMC and 
    // StaticOptimization to solve the muscle force redundancy problem.
    //--------------------------------------------------------------------------    
    virtual osim_double_adouble calcInextensibleTendonActiveFiberForce(SimTK::State& s, 
                                                  osim_double_adouble aActivation) const;
    ///@endcond
//=============================================================================
// PROTECTED METHODS
//=============================================================================
protected:
    struct MuscleLengthInfo;
    struct FiberVelocityInfo;
    struct MuscleDynamicsInfo;
    struct MusclePotentialEnergyInfo;

    /** Developer Access to intermediate values calculate by the muscle model */
    const MuscleLengthInfo& getMuscleLengthInfo(const SimTK::State& s) const;
    MuscleLengthInfo& updMuscleLengthInfo(const SimTK::State& s) const;

    const FiberVelocityInfo& getFiberVelocityInfo(const SimTK::State& s) const;
    FiberVelocityInfo& updFiberVelocityInfo(const SimTK::State& s) const;

    const MuscleDynamicsInfo& getMuscleDynamicsInfo(const SimTK::State& s) const;
    MuscleDynamicsInfo& updMuscleDynamicsInfo(const SimTK::State& s) const;

    const MusclePotentialEnergyInfo& getMusclePotentialEnergyInfo(const SimTK::State& s) const;
    MusclePotentialEnergyInfo& updMusclePotentialEnergyInfo(const SimTK::State& s) const;

    //--------------------------------------------------------------------------
    // CALCULATIONS
    //--------------------------------------------------------------------------
    /** @name Muscle State Dependent Calculations
     *  Developers must override these methods to implement the desired behavior
     *  of their muscle models. Unless you are augmenting the behavior
     *  of an existing muscle class or writing a new derived class, you do not
     *  have access to these methods. 
     */ 
    //@{
    /** calculate muscle's position related values such fiber and tendon lengths,
        normalized lengths, pennation angle, etc... */
    virtual void calcMuscleLengthInfo(const SimTK::State& s, 
        MuscleLengthInfo& mli) const;

    /** calculate muscle's fiber velocity and pennation angular velocity, etc... */
    virtual void calcFiberVelocityInfo(const SimTK::State& s, 
        FiberVelocityInfo& fvi) const;

    /** calculate muscle's active and passive force-length, force-velocity, 
        tendon force, relationships and their related values */
    virtual void  calcMuscleDynamicsInfo(const SimTK::State& s, 
        MuscleDynamicsInfo& mdi) const;

    /** calculate muscle's fiber and tendon potential energy */
    virtual void calcMusclePotentialEnergyInfo(const SimTK::State& s,
        MusclePotentialEnergyInfo& mpei) const;

    /** This function modifies the fiber length in the supplied state such that  
    the fiber and tendon are developing the same force, taking activation and 
    velocity into account. This routine can assume that the state contains a
    meaningful estimate of muscle activation, joint positions, and joint 
    velocities. For example, this can produce fiber lengths suited to 
    beginning a forward dynamics simulation. 
    computeFiberEquilibriumAtZeroVelocity(). */
    virtual void computeInitialFiberEquilibrium(SimTK::State& s) const = 0;

    // End of Muscle's State Related Calculations.
    //@} 

    //--------------------------------------------------------------------------
    // PARENT INTERFACES
    //--------------------------------------------------------------------------
    /** @name Interfaces imposed by parent classes
     */ 
    //@{

    /** Force interface applies tension to bodies, and Muscle also checks that 
        applied muscle tension is not negative. */
    void computeForce(const SimTK::State& state, 
                      SimTK::Vector_<SimTK::SpatialVec>& bodyForces, 
                      SimTK::Vector& generalizedForce) const override;

    /** Potential energy stored by the muscle */
    osim_double_adouble computePotentialEnergy(const SimTK::State& state) const override;

    /** Override PathActuator virtual to calculate a preferred color for the 
    muscle path based on activation. **/
    //SimTK::Vec3 computePathColor(const SimTK::State& state) const override;
    
    /** Model Component creation interface */
    void extendConnectToModel(Model& aModel) override;
    void extendAddToSystem(SimTK::MultibodySystem& system) const override;
    void extendSetPropertiesFromState(const SimTK::State &s) override;
    void extendInitStateFromProperties(SimTK::State& state) const override;
    
    // Update the display geometry attached to the muscle
    virtual void updateGeometry(const SimTK::State& s);
    // End of Interfaces imposed by parent classes.
    //@} 


private:
    void setNull();
    void constructProperties();
    void copyData(const Muscle &aMuscle);

    //--------------------------------------------------------------------------
    // Implement Object interface.
    //--------------------------------------------------------------------------
    /** Override of the default implementation to account for versioning. */
    //void updateFromXMLNode(SimTK::Xml::Element& aNode, int versionNumber=-1) override;


//=============================================================================
// DATA
//=============================================================================
protected:

    /** The assumed fixed muscle-width from which the fiber pennation angle can
        be calculated. */
    osim_double_adouble _muscleWidth;

 /**
    The MuscleLengthInfo struct contains information about the muscle that is
    strictly a function of the length of the fiber and the tendon, and the 
    orientation of the muscle fiber. w.r.t. a fixed orientation of the tendon. 
    
    The function that populates this struct, calcMuscleLengthInfo, is
    called at a point when only the position and orientation of the system are
    known. This function is the first one that is called of the functions
    calcMuscleLengthInfo, calcFiberVelocityInfo and calcMuscleDynamicInfo.
    The velocity and acceleration of the muscle's path will not be known when 
    this function is called.

            NAME                    DIMENSION         UNITS      
             fiberLength              length            m   
             fiberLengthAlongTendon   length            m           [1]
             normFiberLength          length/length     m/m         [2]
                                              
             tendonLength             length            m
             normTendonLength         length/length     m/m         [3]
             tendonStrain             length/length     m/m         [4]
                                                               
             pennationAngle           angle             rad         [5]
             cosPennationAngle        NA                NA          
             sinPennationAngle        NA                NA          
                                         
             fiberPassiveForceLengthMultiplier   force/force     N/N      [6]
             fiberActiveForceLengthMultiplier    force/force     N/N      [7]
        
            userDefinedLengthExtras     NA              NA            [8]

    [1] fiberLengthAlongTendon is the length of the muscle fiber as projected
        on the tendon.

    [2] normFiberLength is the fiberLength normalized with respect to the 
        optimalFiberLength, 

        normFiberLength = fiberLength / optimalFiberLength

        N.B. It is assumed that the optimalFiberLength of a muscle is also 
        its resting length. 
    
    [3] normTendonLength is the tendonLength normalized with respect to the 
        tendonSlackLength

        normTendonLength = tendonLength / tendonSlackLength

    [4] Tendon strain is defined using the elongation of the material divided by 
        its resting length. This is identical to the engineering definition of
        strain. Thus a tendonStrain of 0.01 means that the tendon is currently
        1% longer than its resting length.

        tendonStrain = (tendonLength-tendonSlackLength)/tendonSlackLength
        

    [5] The orientation of the muscle fiber is defined w.r.t. a fixed 
        orientation of the tendon. A pennation angle of 0 means that the fiber 
        is collinear to the tendon. It is normal for the pennation angle
        to range between 0 and Pi/2 radians.

              Fiber                 Tendon
        |===================||-----------------|   Pennation = 0

             ||-------------------|                Pennation = SimTK::Pi/3 
           //                                                   (60 degrees)
          // 
         //  
        //     

    [6] The fiberPassiveForceLengthMultiplier represents the elastic force the fiber 
        generates normalized w.r.t. the maximum isometric force of the fiber.
        Is typically specified by a passiveForceLengthCurve. 
        

    [7] The fiberActiveForceLengthMultiplier is the scaling of the maximum force a fiber 
        can generate as a function of its length. This term usually follows a 
        curve that is zero at a normalized fiber length of 0.5, is 1 at a 
        normalized fiber length of 1, and then zero again at a normalized fiber
        length of 1.5. This curve is generally an interpolation of experimental
        data.

    [8] This vector is left for the muscle modeler to populate with any
        computationally expensive quantities that are computed in 
        calcMuscleLengthInfo, and required for use in the user defined functions 
        calcFiberVelocityInfo and calcMuscleDynamicsInfo. None of the parent 
        classes make any assumptions about what is or isn't in this field 
        - use as necessary.
       
    */
    struct MuscleLengthInfo{             //DIMENSION         Units      
        osim_double_adouble fiberLength;              //length            m  
        osim_double_adouble fiberLengthAlongTendon;   //length            m
        osim_double_adouble normFiberLength;          //length/length     m/m        
                
        osim_double_adouble tendonLength;             //length            m
        osim_double_adouble normTendonLength;         //length/length     m/m        
        osim_double_adouble tendonStrain;             //length/length     m/m        
                                         //
        osim_double_adouble pennationAngle;           //angle             1/s (rads)        
        osim_double_adouble cosPennationAngle;        //NA                NA         
        osim_double_adouble sinPennationAngle;        //NA                NA         

        osim_double_adouble fiberPassiveForceLengthMultiplier;   //NA             NA
        osim_double_adouble fiberActiveForceLengthMultiplier;  //NA             NA
        
        SimTK::Vector userDefinedLengthExtras;//NA        NA

        MuscleLengthInfo(): 
            fiberLength(SimTK::NaN), 
            fiberLengthAlongTendon(SimTK::NaN),
            normFiberLength(SimTK::NaN),            
            tendonLength(SimTK::NaN), 
            normTendonLength(SimTK::NaN), 
            tendonStrain(SimTK::NaN), 
            pennationAngle(SimTK::NaN), 
            cosPennationAngle(SimTK::NaN),
            sinPennationAngle(SimTK::NaN),
            fiberPassiveForceLengthMultiplier(SimTK::NaN), 
            fiberActiveForceLengthMultiplier(SimTK::NaN),
            userDefinedLengthExtras(0, SimTK::NaN){}
        friend std::ostream& operator<<(std::ostream& o, 
            const MuscleLengthInfo& mli) {
            o << "Muscle::MuscleLengthInfo should not be serialized!" 
              << std::endl;
            return o;
        }
    };

    /**
        FiberVelocityInfo contains velocity quantities related to the velocity
        of the muscle (fiber + tendon) complex.
        
        The function that populates this struct, calcFiberVelocityInfo, is called
        when position and velocity information is known. This function is the 
        second function that is called of these related 
        functions:calcMuscleLengthInfo,calcFiberVelocityInfo and 
        calcMuscleDynamicInfo. When calcFiberVelocity is called the acceleration 
        of the muscle path, and any forces the muscle experiences will not be 
        known.

            NAME                     DIMENSION             UNITS
             fiberVelocity             length/time           m/s
             fiberVelocityAlongTendon  length/time           m/s       [1]
             normFiberVelocity         (length/time)/Vmax    NA        [2]
             
             pennationAngularVelocity  angle/time            rad/s     [3]
             
             tendonVelocity            length/time           m/s       
             normTendonVelocity        (length/time)/length  (m/s)/m   [4]
             
             fiberForceVelocityMultiplier force/force          NA        [5]

             userDefinedVelocityExtras    NA                   NA      [6]
        
        [1] fiberVelocityAlongTendon is the first derivative of the symbolic
            equation that defines the fiberLengthAlongTendon.

        [2] normFiberVelocity is the fiberVelocity normalized with respect to 
            the optimal fiber length of the fiber and the maximum fiber velocity
            defined in fiber-lengths/s.

        [3] The sign of the angular velocity is defined using the right 
            hand rule.

        [4] normTendonVelocity is the tendonVelocity (the lengthening velocity 
            of the tendon) divided by its resting length

        [5] The fiberForceVelocityMultiplier is the scaling factor that represents
            how a muscle fiber's force generating capacity is modulated by the
            contraction (concentric or eccentric) velocity of the fiber.
            Generally this curve has a value of 1 at a fiber velocity of 0, 
            has a value of between 1.4-1.8 at the maximum eccentric contraction
            velocity and a value of 0 at the maximum concentric contraction 
            velocity. The force velocity curve, which computes this term,  
            is usually an interpolation of an experimental curve.

        [6] This vector is left for the muscle modeler to populate with any
            computationally expensive quantities that are computed in 
            calcFiberVelocityInfo, and required for use in the user defined 
            function calcMuscleDynamicsInfo. None of the parent classes make 
            any assumptions about what is or isn't in this field
            - use as necessary.

    */
    struct FiberVelocityInfo {              //DIMENSION             UNITS
        osim_double_adouble fiberVelocity;               //length/time           m/s
        osim_double_adouble fiberVelocityAlongTendon;    //length/time           m/s
        osim_double_adouble normFiberVelocity;           //(length/time)/length  (m/s)/m
                                            //
        osim_double_adouble pennationAngularVelocity;    //angle/time            rad/s
                                            //
        osim_double_adouble tendonVelocity;              //length/time           m/s
        osim_double_adouble normTendonVelocity;          //(length/time)/length  (m/s)/m

        osim_double_adouble fiberForceVelocityMultiplier;     //force/force           NA

        SimTK::Vector userDefinedVelocityExtras;//NA                  NA

        FiberVelocityInfo(): 
            fiberVelocity(SimTK::NaN), 
            fiberVelocityAlongTendon(SimTK::NaN),
            normFiberVelocity(SimTK::NaN),
            pennationAngularVelocity(SimTK::NaN),
            tendonVelocity(SimTK::NaN), 
            normTendonVelocity(SimTK::NaN),
            fiberForceVelocityMultiplier(SimTK::NaN),
            userDefinedVelocityExtras(0,SimTK::NaN){};
        friend std::ostream& operator<<(std::ostream& o, 
            const FiberVelocityInfo& fvi) {
            o << "Muscle::FiberVelocityInfo should not be serialized!" 
              << std::endl;
            return o;
        }
    };

    /**
        MuscleDynamicsInfo contains quantities that are related to the forces
        that the muscle generates. 
        
        The function that populates this struct, calcMuscleDynamicsInfo, is 
        called when position and velocity information is known. This function 
        is the last function that is called of these related functions:
        calcMuscleLengthInfo, calcFiberVelocityInfo and calcMuscleDynamicInfo. 


           NAME                         DIMENSION           UNITS                                               
            activation                  NA                   NA     [1]

            fiberForce                  force                N
            fiberForceAlongTendon       force                N      [2]
            normFiberForce              force/force          N/N    [3]
            activeFiberForce            force                N      [4]
            passiveFiberForce           force                N      [5]
                                        
            tendonForce                 force                N
            normTendonForce             force/force          N/N    [6]
                                        
            fiberStiffness              force/length         N/m    [7]   
            fiberStiffnessAlongTendon   force/length         N/m    [8]
            tendonStiffness             force/length         N/m    [9]
            muscleStiffness             force/length         N/m    [10]
                                        
            fiberActivePower            force*velocity       W (N*m/s)
            fiberPassivePower           force*velocity       W (N*m/s)
            tendonPower                 force*velocity       W (N*m/s)
            musclePower                 force*velocity       W (N*m/s)

            userDefinedDynamicsData     NA                   NA     [11]

        [1] This is a quantity that ranges between 0 and 1 that dictates how
            on or activated a muscle is. This term may or may not have its own
            time dependent behavior depending on the muscle model.

        [2] fiberForceAlongTendon is the fraction of the force that is developed
            by the fiber that is transmitted to the tendon. This fraction 
            depends on the pennation model that is used for the muscle model

        [3] This is the force developed by the fiber scaled by the maximum 
            isometric contraction force. Note that the maximum isometric force
            is defined as the maximum isometric force a muscle fiber develops
            at its optimal pennation angle, and along the line of the fiber.

        [4] This is the portion of the fiber force that is created as a direct
            consequence of the value of 'activation'.

        [5] This is the portion of the fiber force that is created by the 
            parallel elastic element within the fiber.
    
        [6] This is the tendonForce normalized by the maximum isometric 
            contraction force

        [7] fiberStiffness is defined as the partial derivative of fiber force
            with respect to fiber length

        [8] fiberStiffnessAlongTendon is defined as the partial derivative of 
            fiber force along the tendon with respect to small changes in
            the fiber length along the tendon. This quantity is normally 
            computed using the equations for fiberStiffness, and then using an 
            application of the chain rule to yield fiberStiffnessAlongTendon.

        [9] tendonStiffness is defined as the partial derivative of tendon
            force with respect to tendon length

        [10] muscleStiffness is defined as the partial derivative of muscle force
            with respect to changes in muscle length. This quantity can usually
            be computed by noting that the tendon and the fiber are in series,
            with the fiber at a pennation angle. Thus

            Kmuscle =   (Kfiber_along_tendon * Ktendon)
                       /(Kfiber_along_tendon + Ktendon) 

        [11] This vector is left for the muscle modeler to populate with any
             computationally expensive quantities that might be of interest 
             after dynamics calculations are completed but maybe of use
             in computing muscle derivatives or reporting values of interest.

    */
    struct MuscleDynamicsInfo {     //DIMENSION             UNITS
        osim_double_adouble activation;              // NA                   NA
                                        //
        osim_double_adouble fiberForce;              // force                N
        osim_double_adouble fiberForceAlongTendon;   // force                N
        osim_double_adouble normFiberForce;          // force/force          N/N
        osim_double_adouble activeFiberForce;        // force                N
        osim_double_adouble passiveFiberForce;       // force                N
                                        //
        osim_double_adouble tendonForce;             // force                N
        osim_double_adouble normTendonForce;         // force/force          N/N
                                        //
        osim_double_adouble fiberStiffness;          // force/length         N/m
        osim_double_adouble fiberStiffnessAlongTendon;//force/length         N/m
        osim_double_adouble tendonStiffness;         // force/length         N/m
        osim_double_adouble muscleStiffness;         // force/length         N/m
                                        //
        osim_double_adouble fiberActivePower;        // force*velocity       W
        osim_double_adouble fiberPassivePower;       // force*velocity       W
        osim_double_adouble tendonPower;             // force*velocity       W
        osim_double_adouble musclePower;             // force*velocity       W

        SimTK::Vector userDefinedDynamicsExtras; //NA          NA

        MuscleDynamicsInfo(): 
            activation(SimTK::NaN), 
            fiberForce(SimTK::NaN),
            fiberForceAlongTendon(SimTK::NaN),
            normFiberForce(SimTK::NaN), 
            activeFiberForce(SimTK::NaN),
            passiveFiberForce(SimTK::NaN),
            tendonForce(SimTK::NaN),
            normTendonForce(SimTK::NaN), 
            fiberStiffness(SimTK::NaN),
            fiberStiffnessAlongTendon(SimTK::NaN),
            tendonStiffness(SimTK::NaN),
            muscleStiffness(SimTK::NaN),
            fiberActivePower(SimTK::NaN),
            fiberPassivePower(SimTK::NaN),
            tendonPower(SimTK::NaN),
            musclePower(SimTK::NaN),
            userDefinedDynamicsExtras(0, SimTK::NaN){};
        friend std::ostream& operator<<(std::ostream& o, 
            const MuscleDynamicsInfo& mdi) {
            o << "Muscle::MuscleDynamicsInfo should not be serialized!" 
              << std::endl;
            return o;
        }
    };

    /**
        MusclePotentialEnergyInfo contains quantities related to the potential
        energy of the muscle (fiber + tendon) complex.
        
        The function that populates this struct, calcMusclePotentialEnergyInfo, can
        be called when position information is known. This function is
        dependent on calcMuscleLengthInfo.

        NAME                     DIMENSION              UNITS
        fiberPotentialEnergy      force*distance         J (Nm)   [1]
        tendonPotentialEnergy     force*distance         J (Nm)   [2]
        musclePotentialEnergy     force*distance         J (Nm)   [3]

        userDefinedPotentialEnergyExtras                         [4]

        [4] This vector is left for the muscle modeler to populate with any
            computationally expensive quantities that are computed in 
            calcMusclePotentialEnergyInfo, that might be useful for others to
            access.

    */
    struct MusclePotentialEnergyInfo {              //DIMENSION             UNITS
        osim_double_adouble fiberPotentialEnergy;     //force*distance    J (Nm)     
        osim_double_adouble tendonPotentialEnergy;    //force*distance    J (Nm)     
        osim_double_adouble musclePotentialEnergy;    //force*distance    J (Nm)

        SimTK::Vector userDefinedPotentialEnergyExtras;//NA                  NA

        MusclePotentialEnergyInfo(): 
            fiberPotentialEnergy(SimTK::NaN),
            tendonPotentialEnergy(SimTK::NaN),
            musclePotentialEnergy(SimTK::NaN), 
            userDefinedPotentialEnergyExtras(0,SimTK::NaN){};
        friend std::ostream& operator<<(std::ostream& o, 
            const MusclePotentialEnergyInfo& fvi) {
            o << "Muscle::MusclePotentialEnergyInfo should not be serialized!" 
              << std::endl;
            return o;
        }
    };


    /** to support deprecated muscles */
    osim_double_adouble _maxIsometricForce;
    osim_double_adouble _optimalFiberLength;
    osim_double_adouble _pennationAngleAtOptimal;
    osim_double_adouble _tendonSlackLength;

//=============================================================================
};  // END of class Muscle
//=============================================================================
//=============================================================================

} // end of namespace OpenSim

#endif // OPENSIM_MUSCLE_H_
