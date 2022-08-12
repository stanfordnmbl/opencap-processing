#ifndef _StateVector_h_
#define _StateVector_h_
/* -------------------------------------------------------------------------- *
 *                          OpenSim:  StateVector.h                           *
 * -------------------------------------------------------------------------- *
 * The OpenSim API is a toolkit for musculoskeletal modeling and simulation.  *
 * See http://opensim.stanford.edu and the NOTICE file for more information.  *
 * OpenSim is developed at Stanford University and supported by the US        *
 * National Institutes of Health (U54 GM072970, R24 HD065690) and by DARPA    *
 * through the Warrior Web program.                                           *
 *                                                                            *
 * Copyright (c) 2005-2017 Stanford University and the Authors                *
 * Author(s): Frank C. Anderson                                               *
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

/* Note: This code was originally developed by Realistic Dynamics Inc. 
 * Author: Frank C. Anderson 
 */


#include "osimCommonDLL.h"
#include "Array.h"

#include "SimTKcommon.h"
#include <vector>

#include "osim_adouble.h"


namespace OpenSim { 

//=============================================================================
//=============================================================================
/**
 * A class which stores a vector of states or data at a specified time.
 * Generally, it is used to store the time histories of the states during
 * an integration.
 *
 * The format of the numerical output for this class is specified by the
 * settings in IO.
 *
 * @author Frank C. Anderson
 * @version 1.0
 * @see IO
 */
class OSIMCOMMON_API StateVector
{
//=============================================================================


//=============================================================================
// DATA
//=============================================================================
private:
    /** Time stamp of the statevector. */
    osim_double_adouble _t;
    /** Array of states. */
    Array<osim_double_adouble> _data;

//=============================================================================
// METHODS
//=============================================================================
public:
    StateVector()                   = default;
    StateVector(const StateVector&) = default;
    virtual ~StateVector();

    StateVector(osim_double_adouble aT);
    StateVector(osim_double_adouble aT, const SimTK::Vector_<osim_double_adouble>& data);

private:
    void setNull();

    //--------------------------------------------------------------------------
    // OPERATORS
    //--------------------------------------------------------------------------
public:
#ifndef SWIG
    StateVector& operator=(const StateVector &aStateVector);
    bool operator==(const StateVector &aStateVector) const;
    bool operator<(const StateVector &aStateVector) const;
    friend std::ostream& operator<<(std::ostream &aOut,
                                    const StateVector &aStateVector) {
        aOut<<"StateVector: t="<<aStateVector._t<<", "<<aStateVector._data;
        return(aOut);
    };
#endif
    //--------------------------------------------------------------------------
    // GET AND SET
    //--------------------------------------------------------------------------
public:
    void setStates(osim_double_adouble aT, const SimTK::Vector_<osim_double_adouble>& data);
    int getSize() const;
    void setTime(osim_double_adouble aT);
	osim_double_adouble  getTime() const;
    int getDataValue(int aIndex, osim_double_adouble &rValue);
    void setDataValue(int aIndex, osim_double_adouble &aValue);
    Array<osim_double_adouble>& getData();
#ifndef SWIG
    const Array<osim_double_adouble>& getData() const;
#endif
    //--------------------------------------------------------------------------
    // OPERATIONS
    //--------------------------------------------------------------------------
    void shiftTime(osim_double_adouble aValue);
    void scaleTime(osim_double_adouble aValue);
    void add(osim_double_adouble aValue);
    void add(const SimTK::Vector_<osim_double_adouble>& values);
    void add(int aN, osim_double_adouble aValue);
    void add(StateVector *aStateVector);
    void subtract(osim_double_adouble aValue);
    void subtract(const SimTK::Vector_<osim_double_adouble>& values);
    void subtract(StateVector *aStateVector);
    void multiply(osim_double_adouble aValue);
    void multiply(const SimTK::Vector_<osim_double_adouble>& values);
    void multiply(StateVector *aStateVector);
    void divide(osim_double_adouble aValue);
    void divide(const SimTK::Vector_<osim_double_adouble>& values);
    void divide(StateVector *aStateVector);

    //--------------------------------------------------------------------------
    // IO
    //--------------------------------------------------------------------------
#ifndef SWIG
    int print(FILE *fp) const;
#endif

//=============================================================================
};  // END of class StateVector

}; //namespace
//=============================================================================
//=============================================================================

#endif //__StateVector_h__
