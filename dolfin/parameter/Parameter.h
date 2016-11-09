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
// Modified by Joachim B Haga 2012
//
// First added:  2009-05-08
// Last changed: 2012-09-11

#ifndef __PARAMETER_H
#define __PARAMETER_H

#include <set>
#include <sstream>
#include <string>

namespace dolfin
{

  /// Base class for parameters.

  class Parameter
  {
  public:

    /// Create parameter for given key
    /// @param key (std::string)
    ///
    explicit Parameter(std::string key);

    /// Destructor
    virtual ~Parameter();

    /// Return parameter key
    /// @return std::string
    std::string key() const;

    /// Return parameter description
    /// @return std::string
    std::string description() const;

    /// Return true if parameter is set, return false otherwise
    /// @return bool
    bool is_set() const;

    /// Reset the parameter to empty, so that is_set() returns false.
    void reset();

    /// Return access count (number of times parameter has been accessed)
    /// @return std::size_t
    std::size_t access_count() const;

    /// Return change count (number of times parameter has been changed)
    /// @return std::size_t
    std::size_t change_count() const;

    /// Set range for int-valued parameter
    /// @param min_value (int)
    /// @param max_value (int)
    ///
    virtual void set_range(int min_value, int max_value);

    /// Set range for double-valued parameter
    /// @param min_value (double)
    /// @param max_value (double)
    ///
    virtual void set_range(double min_value, double max_value);

    /// Set range for string-valued parameter
    /// @param range (std::set<std::string>)
    ///
    virtual void set_range(std::set<std::string> range);

    /// Get range for int-valued parameter
    /// @param [out] min_value (int)
    /// @param [out] max_value (int)
    virtual void get_range(int& min_value, int& max_value) const;

    /// Get range for double-valued parameter
    /// @param [out] min_value (double)
    /// @param [out] max_value (double)
    virtual void get_range(double& min_value, double& max_value) const;

    /// Get range for string-valued parameter
    /// @param [out] range (std::set<std::string>)
    virtual void get_range(std::set<std::string>& range) const;

    /// Assignment from int
    /// @param value (int)
    virtual const Parameter& operator= (int value);

    /// Assignment from double
    /// @param value (double)
    virtual const Parameter& operator= (double value);

    /// Assignment from string
    /// @param value (std::string)
    virtual const Parameter& operator= (std::string value);

    /// Assignment from string
    /// @param value (char *)
    virtual const Parameter& operator= (const char* value);

    /// Assignment from bool
    /// @param value (bool)
    virtual const Parameter& operator= (bool value);

    /// Cast parameter to int
    virtual operator int() const;

    /// Cast parameter to std::size_t
    virtual operator std::size_t() const;

    /// Cast parameter to double
    virtual operator double() const;

    /// Cast parameter to string
    virtual operator std::string() const;

    /// Cast parameter to bool
    virtual operator bool() const;

    /// Return value type string
    virtual std::string type_str() const = 0;

    /// Return value string
    virtual std::string value_str() const = 0;

    /// Return range string
    virtual std::string range_str() const = 0;

    /// Return short string description
    virtual std::string str() const = 0;

    /// Check that key name is allowed
    static void check_key(std::string key);

  protected:

    // Access count
    mutable std::size_t _access_count;

    // Change count
    std::size_t _change_count;

    // Whether or not parameter has been set
    bool _is_set;

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
    explicit IntParameter(std::string key);

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

    /// Cast parameter to std::size_t
    operator std::size_t() const;

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
  class DoubleParameter : public Parameter
  {
  public:

    /// Create unset double-valued parameter
    explicit DoubleParameter(std::string key);

    /// Create double-valued parameter
    DoubleParameter(std::string key, double value);

    /// Destructor
    ~DoubleParameter();

    /// Set range
    void set_range(double min_value, double max_value);

    /// Get range
    void get_range(double &min_value, double &max_value) const;

    /// Assignment
    const DoubleParameter& operator= (double value);

    /// Cast parameter to double
    operator double() const;

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
    double _value;

    /// Parameter range
    double _min, _max;

  };

  /// Parameter with value type string
  class StringParameter : public Parameter
  {
  public:

    /// Create unset string-valued parameter
    explicit StringParameter(std::string key);

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

    /// Create unset bool-valued parameter
    explicit BoolParameter(std::string key);

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
