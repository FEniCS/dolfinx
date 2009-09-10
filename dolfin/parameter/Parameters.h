// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Hake, 2009
// Modified by Garth N. Wells, 2009
//
// First added:  2009-05-08
// Last changed: 2009-09-10

#ifndef __PARAMETERS_H
#define __PARAMETERS_H

#include <set>
#include <map>
#include <vector>
#include "Parameter.h"

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

  class XMLParameters;

  /// This class stores a set of parameters. Each parameter is
  /// identified by a unique string (the key) and a value of some
  /// given value type. Parameter sets can be nested at arbitrary
  /// depths.
  ///
  /// A parameter may be either int, double, string or boolean valued.
  ///
  /// Parameters may be added as follows:
  ///
  ///   Parameters p("my_parameters");
  ///   p.add("relative_tolerance",  1e-15);
  ///   p.add("absolute_tolerance",  1e-15);
  ///   p.add("gmres_restart",       30);
  ///   p.add("monitor_convergence", false);
  ///
  /// Parameters may be changed as follows:
  ///
  ///   p("gmres_restart") = 50;
  ///
  /// Parameter values may be retrieved as follows:
  ///
  ///   int gmres_restart = p("gmres_restart");
  ///
  /// Parameter sets may be nested as follows:
  ///
  ///   Parameters q("nested_parameters");
  ///   p.add(q);
  ///
  /// Nested parameters may then be accessed by
  ///
  ///   p["nested_parameters"]("...")
  ///
  /// Parameters may be nested at arbitrary depths.
  ///
  /// Parameters may be parsed from the command-line as follows:
  ///
  ///   p.parse(argc, argv);
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

    /// Add int-valued parameter
    void add(std::string key, int value);

    /// Add int-valued parameter with given range
    void add(std::string key, int value, int min_value, int max_value);

    /// Add double-valued parameter
    void add(std::string key, double value);

    /// Add double-valued parameter with given range
    void add(std::string key, double value, double min_value, double max_value);

#ifdef HAS_GMP
    /// Add double-valued parameter
    void add(std::string key, real value);

    /// Add double-valued parameter with given range
    void add(std::string key, real value, real min_value, real max_value);
#endif

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

    /// Parse parameters from command-line
    virtual void parse(int argc, char* argv[]);

    /// Update parameters with another set of parameters
    void update(const Parameters& parameters);

    /// Return parameter for given key
    Parameter& operator[] (std::string key);

    /// Return parameter for given key (const version)
    const Parameter& operator[] (std::string key) const;

    // Note: We would have liked to use [] also for access of nested parameter
    // sets just like we do in Python but we can't overload on return type.

    /// Return nested parameter set for given key
    Parameters& operator() (std::string key);

    /// Return nested parameter set for given key (const)
    const Parameters& operator() (std::string key) const;

    /// Assignment operator
    const Parameters& operator= (const Parameters& parameters);

    /// Return a vector of parameter keys
    void get_parameter_keys(std::vector<std::string>& keys) const;

    /// Return a vector of parameter set keys
    void get_parameter_set_keys(std::vector<std::string>& keys) const;

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Define XMLHandler for use in new XML reader/writer
    typedef XMLParameters XMLHandler;

  protected:

    /// Parse filtered options (everything except PETSc options)
    void parse_dolfin(int argc, char* argv[]);

    /// Parse filtered options (only PETSc options)
    void parse_petsc(int argc, char* argv[]);

  private:

    // Add all parameters as options to a boost::program_option instance
    void add_parameter_set_to_po(boost::program_options::options_description& desc,
                                 const Parameters &parameters,
                                 std::string base_name="") const;

    // Read in values from the boost::variable_map
    void read_vm(boost::program_options::variables_map& vm,
                 Parameters &parameters,
                 std::string base_name="");

    // Return pointer to parameter for given key and 0 if not found
    Parameter* find_parameter(std::string key) const;

    // Return pointer to parameter set for given key and 0 if not found
    Parameters* find_parameter_set(std::string key) const;

    // Parameter set key
    std::string _key;

    // Map from key to parameter
    std::map<std::string, Parameter*> _parameters;

    // Map from key to parameter sets
    std::map<std::string, Parameters*> _parameter_sets;

  };

}

#endif
