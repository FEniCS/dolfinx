// Copyright (C) 2009-2012 Anders Logg
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
// Modified by Johan Hake, 2009
// Modified by Garth N. Wells, 2009
//
// First added:  2009-05-08
// Last changed: 2012-10-26

#include <sstream>
#include <stdio.h>
#include <boost/program_options.hpp>
#include <boost/scoped_array.hpp>

#include <dolfin/log/log.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/log/Table.h>
#include <dolfin/common/utils.h>
#include <dolfin/common/SubSystemsManager.h>
#include "Parameter.h"
#include "Parameters.h"

using namespace dolfin;
namespace po = boost::program_options;

// Typedef of iterators for convenience
typedef std::map<std::string, Parameter*>::iterator parameter_iterator;
typedef std::map<std::string, Parameter*>::const_iterator const_parameter_iterator;
typedef std::map<std::string, Parameters*>::iterator parameter_set_iterator;
typedef std::map<std::string, Parameters*>::const_iterator const_parameter_set_iterator;

//-----------------------------------------------------------------------------
Parameters::Parameters(std::string key) : _key(key)
{
  // Check that key name is allowed
  Parameter::check_key(key);
}
//-----------------------------------------------------------------------------
Parameters::~Parameters()
{
  clear();
}
//-----------------------------------------------------------------------------
Parameters::Parameters(const Parameters& parameters)
{
  *this = parameters;
}
//-----------------------------------------------------------------------------
std::string Parameters::name() const
{
  return _key;
}
//-----------------------------------------------------------------------------
void Parameters::rename(std::string key)
{
  Parameter::check_key(key);
  _key = key;
}
//-----------------------------------------------------------------------------
void Parameters::clear()
{
  // Delete parameters
  for (parameter_iterator it = _parameters.begin(); it != _parameters.end();
       ++it)
  {
    delete it->second;
  }
  _parameters.clear();

  // Delete parameter sets
  for (parameter_set_iterator it = _parameter_sets.begin();
       it != _parameter_sets.end(); ++it)
  {
    delete it->second;
  }
  _parameter_sets.clear();

  // Reset key
  _key = "";
}
//-----------------------------------------------------------------------------
void Parameters::add(std::string key, int value)
{
  // Check key name
  if (find_parameter(key))
  {
    dolfin_error("Parameters.cpp",
                 "add parameter",
                 "Parameter \"%s.%s\" already defined",
                 this->name().c_str(), key.c_str());
  }

  // Add parameter
  _parameters[key] = new IntParameter(key, value);
}
//-----------------------------------------------------------------------------
void Parameters::add(std::string key, int value,
                     int min_value, int max_value)
{
  // Add parameter
  add(key, value);

  // Set range
  Parameter* p = find_parameter(key);
  dolfin_assert(p);
  p->set_range(min_value, max_value);
}
//-----------------------------------------------------------------------------
void Parameters::add(std::string key, double value)
{
  // Check key name
  if (find_parameter(key))
  {
    dolfin_error("Parameters.cpp",
                 "add parameter",
                 "Parameter \"%s.%s\" already defined",
                 this->name().c_str(), key.c_str());
  }

  // Add parameter
  _parameters[key] = new DoubleParameter(key, value);
}
//-----------------------------------------------------------------------------
void Parameters::add(std::string key, double value,
                     double min_value, double max_value)
{
  // Add parameter
  add(key, value);

  // Set range
  Parameter* p = find_parameter(key);
  dolfin_assert(p);
  p->set_range(min_value, max_value);
}
//-----------------------------------------------------------------------------
void Parameters::add(std::string key, std::string value)
{
  // Check key name
  if (find_parameter(key))
  {
    dolfin_error("Parameters.cpp",
                 "add parameter",
                 "Parameter \"%s.%s\" already defined",
                 this->name().c_str(), key.c_str());
  }

  // Add parameter
  _parameters[key] = new StringParameter(key, value);
}
//-----------------------------------------------------------------------------
void Parameters::add(std::string key, const char* value)
{
  // This version is needed to avoid having const char* picked up by
  // the add function for bool parameters.

  // Check key name
  if (find_parameter(key))
  {
    dolfin_error("Parameters.cpp",
                 "add parameter",
                 "Parameter \"%s.%s\" already defined",
                 this->name().c_str(), key.c_str());
  }

  // Add parameter
  _parameters[key] = new StringParameter(key, value);
}
//-----------------------------------------------------------------------------
void Parameters::add(std::string key, std::string value,
                     std::set<std::string> range)
{
  // Add parameter
  add(key, value);

  // Set range
  Parameter* p = find_parameter(key);
  dolfin_assert(p);
  p->set_range(range);
}
//-----------------------------------------------------------------------------
void Parameters::add(std::string key, const char* value,
                     std::set<std::string> range)
{
  // This version is needed to avoid having const char* picked up by
  // the add function for bool parameters.

  // Add parameter
  add(key, value);

  // Set range
  Parameter* p = find_parameter(key);
  dolfin_assert(p);
  p->set_range(range);
}
//-----------------------------------------------------------------------------
void Parameters::add(std::string key, bool value)
{
  // Check key name
  if (find_parameter(key))
  {
    dolfin_error("Parameters.cpp",
                 "add parameter",
                 "Parameter \"%s.%s\" already defined",
                 this->name().c_str(), key.c_str());
  }

  // Add parameter
  _parameters[key] = new BoolParameter(key, value);
}
//-----------------------------------------------------------------------------
void Parameters::add(const Parameters& parameters)
{
  // Check key name
  if (find_parameter_set(parameters.name()))
  {
    dolfin_error("Parameters.cpp",
                 "add parameter set",
                 "Parameter set \"%s.%s\" already defined",
                 this->name().c_str(), parameters.name().c_str());
  }

  // Add parameter set
  Parameters* p = new Parameters("");
  *p = parameters;
  _parameter_sets[parameters.name()] = p;
}
//-----------------------------------------------------------------------------
void Parameters::remove(std::string key)
{
  // Check key name
  if (!find_parameter(key) && !find_parameter_set(key))
  {
    dolfin_error("Parameters.cpp",
                 "remove parameter or parameter set",
                 "No parameter or parameter set \"%s.%s\" defined",
                 this->name().c_str(), key.c_str());
  }

  // Delete objects (safe to delete both even if only one is nonzero)
  delete find_parameter(key);
  delete find_parameter_set(key);

  // Remove from maps (safe to remove both)
  std::size_t num_removed = 0;
  num_removed += _parameters.erase(key);
  num_removed += _parameter_sets.erase(key);
  dolfin_assert(num_removed == 1);
}
//-----------------------------------------------------------------------------
void Parameters::parse(int argc, char* argv[])
{
  log(TRACE, "Parsing command-line arguments.");
  parse_common(argc, argv);
}
//-----------------------------------------------------------------------------
void Parameters::update(const Parameters& parameters)
{
  // Update the parameters
  for (const_parameter_iterator it = parameters._parameters.begin();
       it != parameters._parameters.end(); ++it)
  {
    // Get parameters
    const Parameter& other = *it->second;
    Parameter* self = find_parameter(other.key());

    // Skip parameters not in this parameter set (no new parameters added)
    if (!self)
    {
      warning("Ignoring unknown parameter \"%s\" in parameter set \"%s\" when updating parameter set \"%s\".",
              other.key().c_str(), parameters.name().c_str(), name().c_str());
      continue;
    }

    // Skip unset parameters
    if (!other.is_set())
    {
      warning("Ignoring unset parameter \"%s\" in parameter set \"%s\" when updating parameter set \"%s\".",
              other.key().c_str(), parameters.name().c_str(), name().c_str());
      continue;
    }

    // Set value (will give an error if the type is wrong)
    if (other.type_str() == "int")
      *self = static_cast<int>(other);
    else if (other.type_str() == "double")
      *self = static_cast<double>(other);
    else if (other.type_str() == "bool")
      *self = static_cast<bool>(other);
    else if (other.type_str() == "string")
      *self = static_cast<std::string>(other);
    else
    {
      dolfin_error("Parameters.cpp",
                   "update parameter set",
                   "Parameter \"%s\" has unknown type: \"%s\"",
                   other.key().c_str(), other.type_str().c_str());
    }
  }

  // Update nested parameter sets
  for (const_parameter_set_iterator it = parameters._parameter_sets.begin();
       it != parameters._parameter_sets.end(); ++it)
  {
    (*this)(it->first).update(*it->second);
  }
  }
//-----------------------------------------------------------------------------
Parameter& Parameters::operator[] (std::string key)
{
  Parameter* p = find_parameter(key);
  if (!p)
  {
    dolfin_error("Parameters.cpp",
                 "access parameter",
                 "Parameter \"%s.%s\" not defined",
                 this->name().c_str(), key.c_str());
  }

  return *p;
}
//-----------------------------------------------------------------------------
const Parameter& Parameters::operator[] (std::string key) const
{
  Parameter* p = find_parameter(key);
  if (!p)
  {
    dolfin_error("Parameters.cpp",
                 "access parameter",
                 "Parameter \"%s.%s\" not defined",
                 this->name().c_str(), key.c_str());
  }

  return *p;
}
//-----------------------------------------------------------------------------
Parameters& Parameters::operator() (std::string key)
{
  Parameters* p = find_parameter_set(key);
  if (!p)
  {
    dolfin_error("Parameters.cpp",
                 "access parameter set",
                 "Parameter set \"%s.%s\" not defined",
                 this->name().c_str(), key.c_str());
  }

  return *p;
}
//-----------------------------------------------------------------------------
const Parameters& Parameters::operator() (std::string key) const
{
  Parameters* p = find_parameter_set(key);
  if (!p)
  {
    dolfin_error("Parameters.cpp",
                 "access parameter set",
                 "Parameter set \"%s.%s\" not defined",
                 this->name().c_str(), key.c_str());
  }

  return *p;
}
//-----------------------------------------------------------------------------
const Parameters& Parameters::operator= (const Parameters& parameters)
{
  // Clear all parameters
  clear();

  // Note: We're relying on the default copy constructors for the
  // Parameter subclasses here to do their work, which they should
  // since they don't use any dynamically allocated data.

  // Copy key
  _key = parameters._key;

  // Copy parameters
  for (const_parameter_iterator it = parameters._parameters.begin();
       it != parameters._parameters.end(); ++it)
  {
    const Parameter& p = *it->second;
    Parameter* q = 0;
    if (p.type_str() == "int")
      q = new IntParameter(dynamic_cast<const IntParameter&>(p));
    else if (p.type_str() == "double")
      q = new DoubleParameter(dynamic_cast<const DoubleParameter&>(p));
    else if (p.type_str() == "bool")
      q = new BoolParameter(dynamic_cast<const BoolParameter&>(p));
    else if (p.type_str() == "string")
      q = new StringParameter(dynamic_cast<const StringParameter&>(p));
    else
    {
      dolfin_error("Parameters.cpp",
                   "copy parameter set",
                   "Parameter from parameter set \"%s\" to parameter set \"%s\" has unknown type: \"%s\"",
                   parameters.name().c_str(), name().c_str(), p.type_str().c_str());
    }

    _parameters[p.key()] = q;
  }

  // Copy parameter sets
  for (const_parameter_set_iterator it = parameters._parameter_sets.begin();
       it != parameters._parameter_sets.end(); ++it)
  {
    const Parameters& p = *it->second;
    _parameter_sets[p.name()] = new Parameters(p);
  }

  return *this;
}
//-----------------------------------------------------------------------------
bool Parameters::has_key(std::string key) const
{
  return has_parameter(key) || has_parameter_set(key);
}
//-----------------------------------------------------------------------------
bool Parameters::has_parameter(std::string key) const
{
  return find_parameter(key) != 0;
}
//-----------------------------------------------------------------------------
bool Parameters::has_parameter_set(std::string key) const
{
  return find_parameter_set(key) != 0;
}
//-----------------------------------------------------------------------------
void Parameters::get_parameter_keys(std::vector<std::string>& keys) const
{
  keys.reserve(_parameters.size());
  for (const_parameter_iterator it = _parameters.begin();
       it != _parameters.end(); ++it)
  {
    keys.push_back(it->first);
  }
}
//-----------------------------------------------------------------------------
void Parameters::get_parameter_set_keys(std::vector<std::string>& keys) const
{
  keys.reserve(_parameter_sets.size());
  for (const_parameter_set_iterator it = _parameter_sets.begin();
       it != _parameter_sets.end(); ++it)
  {
    keys.push_back(it->first);
  }
 }
//-----------------------------------------------------------------------------
std::string Parameters::str(bool verbose) const
{
  std::stringstream s;
  if (verbose)
  {
    s << str(false) << std::endl << std::endl;
    if (_parameters.empty() && _parameter_sets.empty())
    {
      s << name() << indent("(empty)");
      return s.str();
    }

    Table t(_key);
    for (const_parameter_iterator it = _parameters.begin();
         it != _parameters.end(); ++it)
    {
      Parameter* p = it->second;
      t(p->key(), "type") = p->type_str();
      t(p->key(), "value") = (p->is_set() ? p->value_str() : "<unset>");
      t(p->key(), "range") = p->range_str();
      t(p->key(), "access") = p->access_count();
      t(p->key(), "change") = p->change_count();
    }
    s << indent(t.str(true));

    for (const_parameter_set_iterator it = _parameter_sets.begin();
         it != _parameter_sets.end(); ++it)
    {
      s << "\n\n" << indent(it->second->str(verbose));
    }
  }
  else
  {
    s << "<Parameter set \"" << name() << "\" containing "
      << _parameters.size() << " parameter(s) and "
      << _parameter_sets.size() << " nested parameter set(s)>";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
void Parameters::parse_common(int argc, char* argv[])
{
  // Add list of allowed options to po::options_description
  po::options_description desc("Allowed options");
  add_parameter_set_to_po(desc, *this);

  // Add help option
  desc.add_options()("help", "show help text");

  // Read command-line arguments into po::variables_map
  po::variables_map vm;
  const po::parsed_options parsed
    = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
  po::store(parsed, vm);
  po::notify(vm);

  // Collect and report unrecognized options
  const std::vector<std::string> unrecognized_options
    = po::collect_unrecognized(parsed.options, po::include_positional);
  for (std::size_t i = 0; i < unrecognized_options.size(); i++)
  {
    cout << "Skipping unrecognized option for parameter set \""
         << name() << "\": " << unrecognized_options[i] << endl;
  }

  // Show help text
  if (vm.count("help"))
  {
    std::stringstream s;
    s << desc;
    info(s.str());
    exit(1);
  }

  // Read values from the parsed variable map
  read_vm(vm, *this);
}
//-----------------------------------------------------------------------------
void Parameters::parse_petsc(int argc, char* argv[])
{
  // Return if there are no arguments
  if (argc <= 1)
    return;

  std::string s = "Passing options to PETSc:";
  for (int i = 1; i < argc; i++)
  {
    s.append(" ");
    s.append(std::string(argv[i]));
  }
  info(s);

  SubSystemsManager::init_petsc(argc, argv);
}
//-----------------------------------------------------------------------------
void Parameters::add_parameter_set_to_po(po::options_description& desc,
                                         const Parameters &parameters,
                                         std::string base_name) const
{
  for (const_parameter_iterator it = parameters._parameters.begin();
       it != parameters._parameters.end(); ++it)
  {
    const Parameter& p = *it->second;
    std::string param_name(base_name + p.key());
    if (p.type_str() == "int")
    {
      desc.add_options()(param_name.c_str(), po::value<int>(),
                         p.description().c_str());
    }
    else if (p.type_str() == "bool")
    {
      desc.add_options()(param_name.c_str(), po::value<bool>(),
                         p.description().c_str());
    }
    else if (p.type_str() == "double")
    {
      desc.add_options()(param_name.c_str(), po::value<double>(),
                         p.description().c_str());
    }
    else if (p.type_str() == "string")
    {
      desc.add_options()(param_name.c_str(), po::value<std::string>(),
                         p.description().c_str());
    }
  }

  for (const_parameter_set_iterator it = parameters._parameter_sets.begin();
       it != parameters._parameter_sets.end(); ++it)
  {
    add_parameter_set_to_po(desc, *it->second, base_name + it->first + ".");
  }
}
//-----------------------------------------------------------------------------
void Parameters::read_vm(po::variables_map& vm, Parameters& parameters,
                         std::string base_name)
{
  // Read values from po::variables_map
  for (parameter_iterator it = parameters._parameters.begin();
       it != parameters._parameters.end(); ++it)
  {
    Parameter& p = *it->second;
    std::string param_name(base_name + p.key());
    if (p.type_str() == "int")
    {
      const po::variable_value& v = vm[param_name];
      if (!v.empty())
        p = v.as<int>();
    }
    else if (p.type_str() == "bool")
    {
      const po::variable_value& v = vm[param_name];
      if (!v.empty())
        p = v.as<bool>();
    }
    else if (p.type_str() == "double")
    {
      const po::variable_value& v = vm[param_name];
      if (!v.empty())
        p = v.as<double>();
    }
    else if (p.type_str() == "string")
    {
      const po::variable_value& v = vm[param_name];
      if (!v.empty())
        p = v.as<std::string>();
    }
  }

  for (parameter_set_iterator it = parameters._parameter_sets.begin();
       it != parameters._parameter_sets.end(); ++it)
  {
    read_vm(vm, *it->second, base_name + it->first + ".");
  }
}
//-----------------------------------------------------------------------------
Parameter* Parameters::find_parameter(std::string key) const
{
  const_parameter_iterator p = _parameters.find(key);
  if (p == _parameters.end())
    return 0;
  return p->second;
}
//-----------------------------------------------------------------------------
Parameters* Parameters::find_parameter_set(std::string key) const
{
  const_parameter_set_iterator p = _parameter_sets.find(key);
  if (p == _parameter_sets.end())
    return 0;
  return p->second;
}
//-----------------------------------------------------------------------------
namespace dolfin
{
  Parameters empty_parameters("empty");
}
//-----------------------------------------------------------------------------
