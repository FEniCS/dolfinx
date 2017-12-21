// Copyright (C) 2009-2017 Anders Logg and Garth N. Wells
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

#ifndef __PARAMETERS_H
#define __PARAMETERS_H

#include <map>
#include <set>
#include <vector>
#include <boost/optional.hpp>
#include <boost/variant.hpp>
#include "Parameter.h"
#include <dolfin/log/log.h>

namespace boost
{
  namespace program_options
  {
    class variables_map;
    class options_description;
  }
}

namespace dolfin
{

  /// This class stores a set of parameters. Each parameter is
  /// identified by a unique string (the key) and a value of some
  /// given value type. Parameter sets can be nested at arbitrary
  /// depths.
  ///
  /// A parameter may be either int, double, string or boolean valued.
  ///
  /// Parameters may be added as follows:
  ///
  /// @code{.cpp}
  ///   Parameters p("my_parameters");
  ///   p.add("relative_tolerance",  1e-15);
  ///   p.add("absolute_tolerance",  1e-15);
  ///   p.add("gmres_restart",       30);
  ///   p.add("monitor_convergence", false);
  /// @endcode
  ///
  /// Parameters may be changed as follows:
  ///
  /// @code{.cpp}
  ///   p["gmres_restart"] = 50;
  /// @endcode
  ///
  /// Parameter values may be retrieved as follows:
  ///
  /// @code{.cpp}
  ///   int gmres_restart = p["gmres_restart"];
  /// @endcode
  ///
  /// Parameter sets may be nested as follows:
  ///
  /// @code{.cpp}
  ///   Parameters q("nested_parameters");
  ///   p.add(q);
  /// @endcode
  ///
  /// Nested parameters may then be accessed by
  ///
  /// @code{.cpp}
  ///   p("nested_parameters")["..."]
  /// @endcode
  ///
  /// Parameters may be nested at arbitrary depths.
  ///
  /// Parameters may be parsed from the command-line as follows:
  ///
  /// @code{.cpp}
  ///   p.parse(argc, argv);
  /// @endcode
  ///
  /// Note: spaces in parameter keys are not allowed (to simplify
  /// usage from command-line).

  class Parameters
  {
  public:

    /// Create empty parameter set
    explicit Parameters(std::string key="parameters");

    /// Destructor
    virtual ~Parameters();

    /// Copy constructor
    Parameters(const Parameters& parameters);

    /// Return name for parameter set
    std::string name() const;

    /// Rename parameter set
    void rename(std::string key);

    /// Clear parameter set
    void clear();

    // Note: This is not called 'add' because SWIG does not handle
    // typesafe C++ enums correctly. It may be renamed when switching
    // to pybind11.
    //
    /// Add unset parameter of specified type
    void add_unset(std::string key, Parameter::Type type);

    // Deprecated. Use add_unset (see add_unset note).
    //
    /// Add an unset parameter of type T. For example, to create a
    /// unset parameter of type bool, do
    /// parameters.add<bool>("my_setting")
    template<typename T> void add(std::string key)
    {
      // Check key name
      if (has_parameter(key))
      {
        dolfin_error("Parameters.cpp",
                     "add parameter",
                     "Parameter \"%s.%s\" already defined",
                     this->name().c_str(), key.c_str());
      }

      // Add parameter. Check for bool must come before check for
      // std::is_integral.
      if (std::is_same<T, bool>::value)
        _parameters.insert({key, Parameter(key, Parameter::Type::Bool)});
      else if (std::is_same<T, std::string>::value)
        _parameters.insert({key, Parameter(key, Parameter::Type::String)});
      else if (std::is_integral<T>::value)
        _parameters.insert({key, Parameter(key, Parameter::Type::Int)});
      else if (std::is_floating_point<T>::value)
        _parameters.insert({key, Parameter(key, Parameter::Type::Float)});
      else
      {
        dolfin_error("Parameters.cpp",
                     "add parameter",
                     "Parameter type not supported");
      }
    }

    /// Add an unset parameter of type T with allows parameters. For
    /// example, to create a unset parameter of type bool, do
    /// parameters.add<bool>("my_setting")
    template<typename T> void add(std::string key, T min, T max)
    {
      // Check key name
      if (has_parameter(key))
      {
        dolfin_error("Parameters.cpp",
                     "add parameter",
                     "Parameter \"%s.%s\" already defined",
                     this->name().c_str(), key.c_str());
      }

      // Add parameter
      _parameters.insert({key, Parameter(key, min, max)});
    }

    /// Add an unset parameter of type T with allows parameters. For
    /// example, to create a unset parameter of type bool, do
    /// parameters.add<bool>("my_setting")
    void add(std::string key, std::set<std::string> valid_values)
    {
      // Check key name
      if (has_parameter(key))
      {
        dolfin_error("Parameters.cpp",
                     "add parameter",
                     "Parameter \"%s.%s\" already defined",
                     this->name().c_str(), key.c_str());
      }

      // Add parameter
      _parameters.insert({key, Parameter(key, valid_values)});
    }

    /// Add int-valued parameter
    void add(std::string key, int value);

    /// Add int-valued parameter with given range
    void add(std::string key, int value, int min_value, int max_value);

    /// Add double-valued parameter
    void add(std::string key, double value);

    /// Add double-valued parameter with given range
    void add(std::string key, double value, double min_value, double max_value);

    /// Add string-valued parameter
    void add(std::string key, std::string value);

    /// Add string-valued parameter
    void add(std::string key, const char* value);

    /// Add string-valued parameter with given range
    void add(std::string key, std::string value, std::set<std::string> range);

    /// Add string-valued parameter with given range
    void add(std::string key, const char* value, std::set<std::string> range);

    /// Add bool-valued parameter
    void add(std::string key, bool value);

    /// Add nested parameter set
    void add(const Parameters& parameters);

    /// Remove parameter or parameter set with given key
    void remove(std::string key);

    /// Parse parameters from command-line
    virtual void parse(int argc, char* argv[]);

    /// Update parameters with another set of parameters
    void update(const Parameters& parameters);

    /// Return parameter for given key
    Parameter& operator[] (std::string key);

    /// Return parameter for given key (const version)
    const Parameter& operator[] (std::string key) const;

    // Note: We would have liked to use [] also for access of nested
    // parameter sets just like we do in Python but we can't overload
    // on return type.

    /// Return nested parameter set for given key
    Parameters& operator() (std::string key);

    /// Return nested parameter set for given key (const)
    const Parameters& operator() (std::string key) const;

    /// Assignment operator
    const Parameters& operator= (const Parameters& parameters);

    /// Check if parameter set has key (parameter or nested parameter set)
    bool has_key(std::string key) const;

    /// Check if parameter set has given parameter
    bool has_parameter(std::string key) const;

    /// Check if parameter set has given nested parameter set
    bool has_parameter_set(std::string key) const;

    /// Return a vector of parameter keys
    void get_parameter_keys(std::vector<std::string>& keys) const;

    /// Return a vector of parameter set keys
    void get_parameter_set_keys(std::vector<std::string>& keys) const;

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Return parameter, if present
    boost::optional<Parameter&> find_parameter(std::string key);

    /// Return parameter set, if present
    boost::optional<Parameters&> find_parameter_set(std::string key);

  protected:

    /// Parse filtered options (everything except PETSc options)
    void parse_common(int argc, char* argv[]);

    /// Parse filtered options (only PETSc options)
    void parse_petsc(int argc, char* argv[]);

  private:

    // Add all parameters as options to a boost::program_option
    // instance
    void
      add_parameter_set_to_po(boost::program_options::options_description& desc,
                              const Parameters &parameters,
                              std::string base_name="") const;

    // Read in values from the boost::variable_map
    void read_vm(boost::program_options::variables_map& vm,
                 Parameters &parameters,
                 std::string base_name="");

    // Parameter set key
    std::string _key;

    // Map from key to parameter(s)
    std::map<std::string, boost::variant<Parameter, Parameters>> _parameters;

  public:

    /// Interface for pybind11 iterators
    std::size_t size() const { return  _parameters.size(); }

    /// Interface for pybind11 iterators
    std::map<std::string, boost::variant<Parameter, Parameters>>::const_iterator begin() const
    { return _parameters.cbegin(); }
    //decltype(_parameters.cbegin()) begin() const { return _parameters.cbegin(); }

    /// Interface for pybind11 iterators
    std::map<std::string, boost::variant<Parameter, Parameters>>::const_iterator end() const
    { return _parameters.cend(); }
    //decltype(_parameters.cend()) end() const { return _parameters.cend(); }

  };

  /// Default empty parameters
  extern Parameters empty_parameters;

}

#endif
