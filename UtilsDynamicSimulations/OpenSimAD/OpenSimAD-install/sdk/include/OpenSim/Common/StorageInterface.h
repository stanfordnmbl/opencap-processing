#ifndef _StorageInterface_h_
#define _StorageInterface_h_
/* -------------------------------------------------------------------------- *
 *                        OpenSim:  StorageInterface.h                        *
 * -------------------------------------------------------------------------- *
 * The OpenSim API is a toolkit for musculoskeletal modeling and simulation.  *
 * See http://opensim.stanford.edu and the NOTICE file for more information.  *
 * OpenSim is developed at Stanford University and supported by the US        *
 * National Institutes of Health (U54 GM072970, R24 HD065690) and by DARPA    *
 * through the Warrior Web program.                                           *
 *                                                                            *
 * Copyright (c) 2005-2017 Stanford University and the Authors                *
 * Author(s): Ayman Habib                                                     *
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

/* Abstract an interface out of the Storage class to be used by readers/writers of other file formats 
 * Author: Ayman Habib 
 */

#include "osimCommonDLL.h"
#include "Object.h"
//=============================================================================
//=============================================================================
/**
 *
 * @version 1.0
 * @author Ayman Habib
 */
namespace OpenSim { 

class StateVector;

class OSIMCOMMON_API StorageInterface : public Object {
OpenSim_DECLARE_ABSTRACT_OBJECT(StorageInterface, Object);

//=============================================================================
// METHODS
//=============================================================================
public:
    explicit StorageInterface(const std::string &aFileName) SWIG_DECLARE_EXCEPTION{};
    StorageInterface(const StorageInterface& aStorageInterface) {};
    virtual ~StorageInterface() {}

#ifndef SWIG
    StorageInterface& operator=(const StorageInterface &aStorageInterface)
    {
        Object::operator=(aStorageInterface);
        return(*this);
    }
#endif
    //--------------------------------------------------------------------------
    // GET AND SET
    //--------------------------------------------------------------------------
    // SIZE
    virtual int getSize() const =0;
    // STATEVECTOR
    virtual StateVector* getStateVector(int aTimeIndex) const =0;
    virtual StateVector* getLastStateVector() const =0;
    // TIME
    virtual osim_double_adouble getFirstTime() const =0;
    virtual osim_double_adouble getLastTime() const =0;
    virtual int getTimeColumn(Array<osim_double_adouble>& rTimes,int aStateIndex=-1) const =0;
    virtual void getTimeColumnWithStartTime(Array<osim_double_adouble>& rTimes,osim_double_adouble startTime=0.0) const =0;
    // DATA
    virtual int getDataAtTime(osim_double_adouble aTime,int aN,Array<osim_double_adouble> &rData) const =0;
    virtual void getDataColumn(const std::string& columnName, Array<osim_double_adouble>& data, osim_double_adouble startTime=0.0) =0;

    //--------------------------------------------------------------------------
    // STORAGE
    //--------------------------------------------------------------------------
    virtual int append(const StateVector &aVec, bool aCheckForDuplicateTime=true) =0;
    virtual int append(const Array<StateVector> &aArray) =0;
    virtual int append(osim_double_adouble aT,int aN,const osim_double_adouble *aY, bool aCheckForDuplicateTime=true) =0;
    virtual int append(osim_double_adouble aT,const SimTK::Vector& aY, bool aCheckForDuplicateTime=true) =0;
    virtual int append(osim_double_adouble aT, const SimTK::Vec3& aY,bool aCheckForDuplicateTime=true){
        return append(aT, 3, &aY[0], aCheckForDuplicateTime);
    }
    virtual int store(int aStep,osim_double_adouble aT,int aN,const osim_double_adouble *aY) =0;

    //--------------------------------------------------------------------------
    // UTILITY
    //--------------------------------------------------------------------------
    virtual int findIndex(osim_double_adouble aT) const =0;
    virtual int findIndex(int aI,osim_double_adouble aT) const =0;
    //--------------------------------------------------------------------------
    // IO
    //--------------------------------------------------------------------------
    virtual void setOutputFileName(const std::string& aFileName) =0;

//=============================================================================
};  // END of class StorageInterface

}; //namespace
//=============================================================================
//=============================================================================

#endif //__StorageInterface_h__
