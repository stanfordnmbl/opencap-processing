//#ifndef OPENSIM_PIECEWISE_CONSTANT_FUNCTION_H_
//#define OPENSIM_PIECEWISE_CONSTANT_FUNCTION_H_
///* -------------------------------------------------------------------------- *
// *                   OpenSim:  PiecewiseConstantFunction.h                    *
// * -------------------------------------------------------------------------- *
// * The OpenSim API is a toolkit for musculoskeletal modeling and simulation.  *
// * See http://opensim.stanford.edu and the NOTICE file for more information.  *
// * OpenSim is developed at Stanford University and supported by the US        *
// * National Institutes of Health (U54 GM072970, R24 HD065690) and by DARPA    *
// * through the Warrior Web program.                                           *
// *                                                                            *
// * Copyright (c) 2005-2017 Stanford University and the Authors                *
// * Author(s): Peter Loan                                                      *
// *                                                                            *
// * Licensed under the Apache License, Version 2.0 (the "License"); you may    *
// * not use this file except in compliance with the License. You may obtain a  *
// * copy of the License at http://www.apache.org/licenses/LICENSE-2.0.         *
// *                                                                            *
// * Unless required by applicable law or agreed to in writing, software        *
// * distributed under the License is distributed on an "AS IS" BASIS,          *
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
// * See the License for the specific language governing permissions and        *
// * limitations under the License.                                             *
// * -------------------------------------------------------------------------- */
//
//
//// INCLUDES
//#include "osimCommonDLL.h"
//#include <string>
//#include "Function.h"
//#include "PropertyDblArray.h"
//
//
////=============================================================================
////=============================================================================
//namespace OpenSim { 
//
//template <class T> class Array;
//
///**
// * A class implementing a step function.
// *
// * This class inherits from Function and so can be used as input to
// * any class requiring a Function as input.
// *
// * @author Peter Loan
// * @version 1.0
// */
//class OSIMCOMMON_API PiecewiseConstantFunction : public Function {
//OpenSim_DECLARE_CONCRETE_OBJECT(PiecewiseConstantFunction, Function);
//
////=============================================================================
//// MEMBER VARIABLES
////=============================================================================
//protected:
//       // PROPERTIES
//       /** Array of values for the independent variables (i.e., the knot
//       sequence).  This array must be monotonically increasing. */
//       PropertyDblArray _propX;
//       Array<osim_double_adouble> &_x;
//
//       /** Y values. */
//       PropertyDblArray _propY;
//       Array<osim_double_adouble> &_y;
//
////=============================================================================
//// METHODS
////=============================================================================
//public:
//    //--------------------------------------------------------------------------
//    // CONSTRUCTION
//    //--------------------------------------------------------------------------
//    PiecewiseConstantFunction();
//    PiecewiseConstantFunction(int aN,const osim_double_adouble *aTimes,const osim_double_adouble *aValues,
//            const std::string &aName="");
//    PiecewiseConstantFunction(const PiecewiseConstantFunction &aFunction);
//    virtual ~PiecewiseConstantFunction();
//
//    void init(Function* aFunction) override;
//
//private:
//    void setNull();
//    void setupProperties();
//    void setEqual(const PiecewiseConstantFunction &aFunction);
//
//    //--------------------------------------------------------------------------
//    // OPERATORS
//    //--------------------------------------------------------------------------
//public:
//#ifndef SWIG
//    PiecewiseConstantFunction& operator=(const PiecewiseConstantFunction &aFunction);
//#endif
//    //--------------------------------------------------------------------------
//    // SET AND GET
//    //--------------------------------------------------------------------------
//public:
//    int getSize() const;
//    const Array<osim_double_adouble>& getX() const;
//    const Array<osim_double_adouble>& getY() const;
//    virtual const osim_double_adouble* getXValues() const;
//    virtual const osim_double_adouble* getYValues() const;
//    virtual int getNumberOfPoints() const { return _x.getSize(); }
//    virtual osim_double_adouble getX(int aIndex) const;
//    virtual osim_double_adouble getY(int aIndex) const;
//    virtual osim_double_adouble getZ(int aIndex) const { return 0.0; }
//    virtual void setX(int aIndex, osim_double_adouble aValue);
//    virtual void setY(int aIndex, osim_double_adouble aValue);
//    virtual bool deletePoint(int aIndex);
//    virtual bool deletePoints(const Array<int>& indices);
//    virtual int addPoint(osim_double_adouble aX, osim_double_adouble aY);
//
//    //--------------------------------------------------------------------------
//    // EVALUATION
//    //--------------------------------------------------------------------------
//    virtual osim_double_adouble evaluateTotalFirstDerivative(osim_double_adouble aX, osim_double_adouble aDxdt) const;
//    virtual osim_double_adouble evaluateTotalSecondDerivative(osim_double_adouble aX, osim_double_adouble aDxdt, osim_double_adouble aD2xdt2) const;
//	osim_double_adouble calcValue(const SimTK::Vector& x) const override;
//	osim_double_adouble calcDerivative(const std::vector<int>& derivComponents, const SimTK::Vector& x) const override;
//    int getArgumentSize() const override;
//    int getMaxDerivativeOrder() const override;
//    SimTK::Function* createSimTKFunction() const override;
//
////=============================================================================
//};     // END class PiecewiseConstantFunction
//
//}; //namespace
////=============================================================================
////=============================================================================
//
//#endif // OPENSIM_PIECEWISE_CONSTANT_FUNCTION_H_
//
