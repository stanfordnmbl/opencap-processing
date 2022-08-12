#ifndef Recorder_H_
#define Recorder_H_

/*
 *      Recorder -- A package for Algorithmic Differentiation with CasADi
 *
 *      Copyright (C) 2019 The Authors
 *      Author: Joris Gillis
 *      Contributor: Antoine Falisse
 *
 *      Licensed under the Apache License, Version 2.0 (the "License"); you
 *      may not use this file except in compliance with the License. You may
 *      obtain a copy of the License at
 *      http://www.apache.org/licenses/LICENSE-2.0.
 *
 *      Unless required by applicable law or agreed to in writing, software
 *      distributed under the License is distributed on an "AS IS" BASIS,
 *      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 *      implied. See the License for the specific language governing
 *      permissions and limitations under the License.
 */

#include <iostream>

#if defined _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif // defined _WIN32

class DLL_EXPORT Recorder {
public:
  ~Recorder();
  Recorder();
  Recorder(double value);
  void operator<<=(double value);
  void operator>>=(double& value);
  explicit operator bool() const;
  Recorder(const Recorder& r);
  friend DLL_EXPORT std::ostream& operator<<(std::ostream &stream, const Recorder& obj);
  static void stop_recording();

  /* Assignments */
  double getValue() const;
  inline double value() const {return getValue();}
  inline Recorder& operator = ( double arg) { return operator=(Recorder(arg)); }
  Recorder& operator = ( const Recorder& );

  /* IO friends */
  friend DLL_EXPORT std::istream& operator >> (std::istream& is, const Recorder& a);

  /* Operation and assignment */
  inline Recorder& operator += ( double value ) { return operator+=(Recorder(value)); }
  inline Recorder& operator += ( const Recorder& value) { return operator=(*this+value); }
  inline Recorder& operator -= ( double value ) { return operator-=(Recorder(value)); }
  inline Recorder& operator -= ( const Recorder& value) { return operator=(*this-value); }
  inline Recorder& operator *= ( double value)  { return operator*=(Recorder(value)); }
  inline Recorder& operator *= ( const Recorder& value) { return operator=(*this*value); }
  inline Recorder& operator /= ( double value)  { return operator/=(Recorder(value)); }
  inline Recorder& operator /= ( const Recorder& value) { return operator=(*this/value); }

  /* Comparison */
  friend bool DLL_EXPORT operator != ( const Recorder&, const Recorder& );
  friend bool DLL_EXPORT operator == ( const Recorder&, const Recorder& );
  friend bool DLL_EXPORT operator <= ( const Recorder&, const Recorder& );
  friend bool DLL_EXPORT operator >= ( const Recorder&, const Recorder& );
  friend bool DLL_EXPORT operator >  ( const Recorder&, const Recorder& );
  friend bool DLL_EXPORT operator <  ( const Recorder&, const Recorder& );
  inline friend bool operator != (double lhs, const Recorder& rhs) { return Recorder(lhs)!=rhs; }
  inline friend bool operator == ( double lhs, const Recorder& rhs) { return Recorder(lhs)==rhs; }
  inline friend bool operator <= ( double lhs, const Recorder& rhs) { return Recorder(lhs)<=rhs; }
  inline friend bool operator >= ( double lhs, const Recorder& rhs) { return Recorder(lhs)>=rhs; }
  inline friend bool operator >  ( double lhs, const Recorder& rhs) { return Recorder(lhs)>rhs; }
  inline friend bool operator <  ( double lhs, const Recorder& rhs) { return Recorder(lhs)<rhs; }

  /* Sign operators */
  inline friend Recorder operator + ( const Recorder& x ) { return x; }
  friend Recorder DLL_EXPORT  operator - ( const Recorder& x );

  /* Binary operators */
  friend Recorder DLL_EXPORT operator + ( const Recorder&, const Recorder& );
  inline friend Recorder operator + ( double lhs, const Recorder& rhs) { return Recorder(lhs)+rhs; }
  inline friend Recorder operator + ( const Recorder& lhs, double rhs)  { return lhs+Recorder(rhs); }
  friend DLL_EXPORT Recorder operator - ( const Recorder&, const Recorder& );
  inline friend Recorder operator - ( const Recorder& lhs, double rhs ) { return lhs-Recorder(rhs); }
  inline friend Recorder operator - ( double lhs, const Recorder& rhs )  { return Recorder(lhs)-rhs; }
  friend DLL_EXPORT Recorder operator * ( const Recorder&, const Recorder& );
  inline friend Recorder operator * ( double lhs, const Recorder& rhs)  { return Recorder(lhs)*rhs; }
  inline friend Recorder operator * ( const Recorder& lhs, double rhs) { return lhs*Recorder(rhs); }
  inline friend Recorder operator / ( const Recorder& lhs, double rhs) { return lhs/Recorder(rhs); }
  friend DLL_EXPORT Recorder operator / ( const Recorder&, const Recorder& );
  friend Recorder operator / ( double lhs, const Recorder& rhs )  { return Recorder(lhs)/rhs; }

  /* Unary operators */
  friend Recorder DLL_EXPORT exp  ( const Recorder& );
  friend Recorder DLL_EXPORT log  ( const Recorder& );
  friend Recorder DLL_EXPORT sqrt ( const Recorder& );
  friend Recorder DLL_EXPORT sin  ( const Recorder& );
  friend Recorder DLL_EXPORT cos  ( const Recorder& );
  friend Recorder DLL_EXPORT tan  ( const Recorder& );
  friend Recorder DLL_EXPORT asin ( const Recorder& );
  friend Recorder DLL_EXPORT acos ( const Recorder& );
  friend Recorder DLL_EXPORT atan ( const Recorder& );

  /* Additional functions */
  friend Recorder DLL_EXPORT sinh  ( const Recorder& );
  friend Recorder DLL_EXPORT cosh  ( const Recorder& );
  friend Recorder DLL_EXPORT tanh  ( const Recorder& );
  friend Recorder DLL_EXPORT asinh ( const Recorder& );
  friend Recorder DLL_EXPORT acosh ( const Recorder& );
  friend Recorder DLL_EXPORT atanh ( const Recorder& );
  friend Recorder DLL_EXPORT erf   ( const Recorder& );
  friend Recorder DLL_EXPORT fabs  ( const Recorder& );
  friend Recorder DLL_EXPORT ceil  ( const Recorder& );
  friend Recorder DLL_EXPORT floor ( const Recorder& );
  friend Recorder DLL_EXPORT fmax ( const Recorder&, const Recorder& );
  inline friend Recorder fmax ( double lhs, const Recorder& rhs) { return fmax(Recorder(lhs), rhs); }
  inline friend Recorder fmax ( const Recorder& lhs, double rhs) { return fmax(lhs, Recorder(rhs)); }
  friend Recorder DLL_EXPORT fmin ( const Recorder&, const Recorder& );
  inline friend Recorder fmin ( double lhs, const Recorder& rhs) { return fmin(Recorder(lhs), rhs); }
  inline friend Recorder fmin ( const Recorder& lhs, double rhs) { return fmin(lhs, Recorder(rhs)); }

  /* Special operators */
  friend Recorder DLL_EXPORT atan2 ( const Recorder&, const Recorder& );
  friend DLL_EXPORT Recorder log10 ( const Recorder& );
  friend Recorder DLL_EXPORT pow ( const Recorder&, const Recorder& );
  inline friend Recorder pow ( double lhs, const Recorder& rhs) { return pow(Recorder(lhs), rhs); }
  inline friend Recorder pow ( const Recorder& lhs, double rhs) { return pow(lhs, Recorder(rhs)); }

protected:
  void disp(std::ostream &stream) const;
  static int get_id();
  bool is_symbol() const;
  std::string repr() const;
  static Recorder from_binary(const Recorder& lhs, const Recorder& rhs, double res, const std::string& op);
  static Recorder from_unary(const Recorder& arg, double res, const std::string& op);

  static std::ofstream& stream();
  explicit Recorder(double value, int id);
  double value_;
  int id_;

  static int counter;
  static int counter_input;
  static int counter_output;
  static int counter_bool;
};

#endif // Recorder_H_
