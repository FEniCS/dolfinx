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

#ifndef __PARAMETER_H
#define __PARAMETER_H

#include <array>
#include <set>
#include <sstream>
#include <string>
#include <boost/blank.hpp>
#include <boost/variant.hpp>

namespace dolfin
{

  /// Base class for parameters.

  class Parameter
  {
  public:

    /// Create parameter for given key and value
    /// @param key (std::string)
    /// @param x (T)
    template<typename T>
      Parameter(std::string key, T x) : _value(x), _access_count(0),
      _change_count(0), _is_set(true), _key(key),
      _description("missing description") { check_key(key); }

    /// Create parameter for given key and value.  This verison (const
    /// char*) is necessary to have the parameter treated as a string
    /// rather than char* being cast as bool.
    ///
    /// @param key (std::string)
    /// @param x (const char*))
    Parameter(std::string key, const char* x);

    /// Enum for the parameter type
    enum class Type { Bool, Int, Float, String };

    /// Create an unset parameter (type is specified, value is
    /// unknown)
    ///
    /// @param key (std::string)
    /// @param ptype (Type))
    Parameter(std::string key, Type ptype);

    /// Create and unset numerical parameter with specified (min, max)
    /// range
    template<typename T>
      Parameter(std::string key, T min, T max) : _value(T(0)),
      _range(std::array<T, 2>({{min, max}})), _access_count(0), _change_count(0),
      _is_set(false), _key(key), _description("missing description")
    { check_key(key); }

    /// Create and unset string parameter with set of allowable strings
  Parameter(std::string key, std::set<std::string> range) :  _value(std::string("")),
      _range(range), _access_count(0), _change_count(0), _is_set(false),
      _key(key), _description("missing description") { check_key(key); }

    /// Copy constructor
    Parameter(const Parameter&) = default;

    /// Move constructor
    Parameter(Parameter&&) = default;

    /// Destructor
    virtual ~Parameter();

    /// Assignment operator
    Parameter& operator= (const Parameter &) = default;

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
    void set_range(int min_value, int max_value);

    /// Set range for double-valued parameter
    /// @param min_value (double)
    /// @param max_value (double)
    ///
    void set_range(double min_value, double max_value);

    /// Set range for string-valued parameter
    /// @param range (std::set<std::string>)
    ///
    void set_range(std::set<std::string> range);

    /// Get range for int-valued parameter
    /// @param [out] min_value (int)
    /// @param [out] max_value (int)
    void get_range(int& min_value, int& max_value) const;

    /// Get range for double-valued parameter
    /// @param [out] min_value (double)
    /// @param [out] max_value (double)
    void get_range(double& min_value, double& max_value) const;

    /// Get range for string-valued parameter
    /// @param [out] range (std::set<std::string>)
    void get_range(std::set<std::string>& range) const;

    /// Assignment from int
    /// @param value (int)
    const Parameter& operator= (int value);

    /// Assignment from double
    /// @param value (double)
    const Parameter& operator= (double value);

    /// Assignment from string
    /// @param value (std::string)
    const Parameter& operator= (std::string value);

    /// Assignment from string
    /// @param value (char *)
    const Parameter& operator= (const char* value);

    /// Assignment from bool
    /// @param value (bool)
    const Parameter& operator= (bool value);

    /// Return parameter value
    boost::variant<boost::blank, bool, int, double, std::string> value() const;

    /// Cast parameter to int
    operator int() const;

    /// Cast parameter to std::size_t
    operator std::size_t() const;

    /// Cast parameter to double
    operator double() const;

    /// Cast parameter to string
    operator std::string() const;

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

    /// Check that key name is allowed
    static void check_key(std::string key);

  protected:

    // Value (0: blank, 1: bool, 2: int, 3: double, 4: std::string)
    boost::variant<boost::blank, bool, int, double, std::string> _value;

    // Ranges (0: blank, 1: int, 2: double, 3: std::string)
    boost::variant<boost::blank, std::array<int, 2>, std::array<double, 2>,
      std::set<std::string>> _range;

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
}
#endif
