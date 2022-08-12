#ifndef OPENSIM_STEP_FUNCTION_H_
#define OPENSIM_STEP_FUNCTION_H_
/* -------------------------------------------------------------------------- *
 *                          OpenSim:  StepFunction.h                          *
 * -------------------------------------------------------------------------- *
 * The OpenSim API is a toolkit for musculoskeletal modeling and simulation.  *
 * See http://opensim.stanford.edu and the NOTICE file for more information.  *
 * OpenSim is developed at Stanford University and supported by the US        *
 * National Institutes of Health (U54 GM072970, R24 HD065690) and by DARPA    *
 * through the Warrior Web program.                                           *
 *                                                                            *
 * Copyright (c) 2005-2017 Stanford University and the Authors                *
 * Author(s): Ajay Seth                                                       *
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


// INCLUDES
#include "Function.h"
#include "PropertyDbl.h"

namespace OpenSim {

//=============================================================================
//=============================================================================
/**
 * A class for representing a StepFunction.
 *
 *          {   start_value,    t <= start_time 
 * f(t) =   {   S-polynomial(t), start_time < t < end_time
 *          {   end_value,      t >= end_time
 *
 * This class inherits from Function and so can be used as input to
 * any class requiring a Function as input.
 *
 * @author Ajay Seth
 * @version 1.0
 */
class OSIMCOMMON_API StepFunction : public Function {
OpenSim_DECLARE_CONCRETE_OBJECT(StepFunction, Function);

//=============================================================================
// MEMBER VARIABLES
//=============================================================================
protected:
    PropertyDbl _startTimeProp;
    osim_double_adouble &_startTime;

    PropertyDbl _endTimeProp;
    osim_double_adouble &_endTime;

    PropertyDbl _startValueProp;
    osim_double_adouble &_startValue;

    PropertyDbl _endValueProp;
    osim_double_adouble &_endValue;

//=============================================================================
// METHODS
//=============================================================================
public:
    //--------------------------------------------------------------------------
    // CONSTRUCTION
    //--------------------------------------------------------------------------
    StepFunction();
    StepFunction(osim_double_adouble startTime, osim_double_adouble endTime, osim_double_adouble startValue=0.0, osim_double_adouble endValue=1.0);
    StepFunction(const StepFunction &aSpline);
    virtual ~StepFunction();

private:
    void setNull();
    void setupProperties();
    void copyData(const StepFunction &aStepFunction);

    //--------------------------------------------------------------------------
    // OPERATORS
    //--------------------------------------------------------------------------
public:
#ifndef SWIG
    StepFunction& operator=(const StepFunction &aStepFunction);
#endif

    //--------------------------------------------------------------------------
    // SET AND GET Coefficients
    //--------------------------------------------------------------------------
public:
    /** %Set step transition start time */
    void setStartTime(osim_double_adouble time)
        { _startTime = time; };
    /** Get step transition time */
    osim_double_adouble getStartTime() const
        { return _startTime; };

    /** %Set step transition end time */
    void setEndTime(osim_double_adouble time)
        { _endTime = time; };
    /** Get step transition time */
    osim_double_adouble getEndTime() const
        { return _endTime; };

    /** %Set start value before step */
    void setStartValue(osim_double_adouble start)
        { _startValue = start; };
    /** Get start value before step */
    osim_double_adouble getStartValue() const
        { return _startValue; };

    /** %Set end value before step */
    void setEndValue(osim_double_adouble end)
        { _endValue = end; };
    /** Get end value before step */
    osim_double_adouble getEndValue() const
        { return _endValue; };

    //--------------------------------------------------------------------------
    // EVALUATION
    //--------------------------------------------------------------------------
    SimTK::Function* createSimTKFunction() const override;

//=============================================================================
};  // END class StepFunction
//=============================================================================
//=============================================================================

} // end of namespace OpenSim

#endif  // OPENSIM_STEP_FUNCTION_H_
