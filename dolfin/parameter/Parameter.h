// Copyright (C) 2009 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2009-05-08
// Last changed: 2009-10-12

#ifndef __PARAMETER_H
#define __PARAMETER_H

#include <set>
#include <sstream>
#include <string>
#include <dolfin/common/types.h>
#include <dolfin/common/real.h>

namespace dolfin
{

  /// Base class for parameters.

  class Parameter
  {
  public:

    /// Create parameter for given key
    Parameter(std::string key);

    /// Destructor
    virtual ~Parameter();

    /// Return parameter key
    std::string key() const;

    /// Return parameter description
    std::string description() const;

    /// Return true if parameter is set, return false otherwise
    bool is_set() const;

    /// Return access count (number of times parameter has been accessed)
    uint access_count() const;

    /// Return change count (number of times parameter has been changed)
    uint change_count() const;

    /// Set range for int-valued parameter
    virtual void set_range(int min_value, int max_value);

    /// Set range for double-valued parameter
    virtual void set_range(real min_value, real max_value);

    /// Set range for string-valued parameter
    virtual void set_range(std::set<std::string> range);

    /// Get range for int-valued parameter
    virtual void get_range(int& min_value, int& max_value) const;

    /// Get range for double-valued parameter
    virtual void get_range(real& min_value, real& max_value) const;

    /// Get range for string-valued parameter
    virtual void get_range(std::set<std::string>& range) const;

    /// Assignment from int
    virtual const Parameter& operator= (int value);

    /// Assignment from double
    virtual const Parameter& operator= (double value);

#ifdef HAS_GMP
    /// Assignment from GMP type
    virtual const Parameter& operator= (real value);
#endif

    /// Assignment from string
    virtual const Parameter& operator= (std::string value);

    /// Assignment from string
    virtual const Parameter& operator= (const char* value);

    /// Assignment from bool
    virtual const Parameter& operator= (bool value);

    /// Cast parameter to int
    virtual operator int() const;

    /// Cast parameter to uint
    virtual operator dolfin::uint() const;

    /// Cast parameter to double
    virtual operator double() const;

    /// Cast parameter to string
    virtual operator std::string() const;

    /// Cast parameter to bool
    virtual operator bool() const;

    /// Get real value of parameter with (possibly) extended precision
    virtual real get_real() const;

    /// Return value type string
    virtual std::string type_str() const = 0;

    /// Return value string
    virtual std::string value_str() const = 0;

    /// Return range string
    virtual std::string range_str() const = 0;

    /// Return short string description
    virtual std::string str() const = 0;

    // Check that key name is allowed
    static void check_key(std::string key);

  protected:

    // Access count
    mutable uint _access_count;

    // Change count
    uint _change_count;

    // Whether or not parameter has been set
    bool _is_set;

  private:

    // Parameter key
    std::string _key;

    // Parameter description
    std::string _description;

  };

  /// Parameter with value type int
  class IntParameter : public Parameter
  {
  public:

    /// Create unset int-valued
    IntParameter(std::string key);

    /// Create int-valued parameter
    IntParameter(std::string key, int value);

    /// Destructor
    ~IntParameter();

    /// Set range
    void set_range(int min_value, int max_value);

    /// Get range
    void get_range(int &min_value, int &max_value) const;

    /// Assignment
    const IntParameter& operator= (int value);

    /// Cast parameter to int
    operator int() const;

    /// Cast parameter to uint
    operator dolfin::uint() const;

    /// Return value type string
    std::string type_str() const;

    /// Return value string
    std::string value_str() const;

    /// Return range string
    std::string range_str() const;

    /// Return short string description
    std::string str() const;

  private:

    /// Parameter value
    int _value;

    /// Parameter range
    int _min, _max;

  };

  /// Parameter with value type double
  class RealParameter : public Parameter
  {
  public:

    /// Create unset double-valued parameter
    RealParameter(std::string key);

    /// Create double-valued parameter
    RealParameter(std::string key, real value);

    /// Destructor
    ~RealParameter();

    /// Set range
    void set_range(real min_value, real max_value);

    /// Get range
    void get_range(real &min_value, real &max_value) const;

    /// Assignment
    const RealParameter& operator= (double value);
#ifdef HAS_GMP
    const RealParameter& operator= (real value);
#endif

    /// Cast parameter to double
    operator double() const;

    /// Get real value (possibly with extended precision)
    real get_real() const;

    /// Return value type string
    std::string type_str() const;

    /// Return value string
    std::string value_str() const;

    /// Return range string
    std::string range_str() const;

    /// Return short string description
    std::string str() const;

  private:

    /// Parameter value
    real _value;

    /// Parameter range
    real _min, _max;

  };

  /// Parameter with value type string
  class StringParameter : public Parameter
  {
  public:

    /// Create unset string-valued parameter
    StringParameter(std::string key);

    /// Create string-valued parameter
    StringParameter(std::string key, std::string value);

    /// Destructor
    ~StringParameter();

    /// Set range
    void set_range(std::set<std::string> range);

    /// Get range
    void get_range(std::set<std::string>& range) const;

    /// Assignment
    const StringParameter& operator= (std::string value);

    /// Assignment
    const StringParameter& operator= (const char* value);

    /// Cast parameter to string
    operator std::string() const;

    /// Return value type string
    std::string type_str() const;

    /// Return value string
    std::string value_str() const;

    /// Return range string
    std::string range_str() const;

    /// Return short string description
    std::string str() const;

  private:

    /// Parameter value
    std::string _value;

    /// Parameter range
    std::set<std::string> _range;

  };

  /// Parameter with value type bool
  class BoolParameter : public Parameter
  {
  public:

    /// Create usnet bool-valued parameter
    BoolParameter(std::string key);

    /// Create bool-valued parameter
    BoolParameter(std::string key, bool value);

    /// Destructor
    ~BoolParameter();

    /// Assignment
    const BoolParameter& operator= (bool value);

    /// Cast parameter to bool
    operator bool() const;

    /// Return value type string
    std::string type_str() const;

    /// Return value string
    std::string value_str() const;

    /// Return range string
    std::string range_str() const;

    /// Return short string description
    std::string str() const;

  private:

    /// Parameter value
    bool _value;

  };

}

#endif
