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
  // Delete parameters and parameter sets
  _parameters.clear();
  _parameter_sets.clear();

  // Reset key
  _key = "";
}
//-----------------------------------------------------------------------------
void Parameters::add_unset(std::string key, Parameter::Type type)
{
  auto e = _parameters.insert({key, Parameter(key, type)});
  if (!e.second)
  {
    dolfin_error("Parameters.cpp",
                 "add parameter",
                 "Parameter \"%s.%s\" already defined",
                 this->name().c_str(), key.c_str());
  }
}
//-----------------------------------------------------------------------------
void Parameters::add(std::string key, int value)
{
  auto e = _parameters.insert({key, Parameter(key, value)});
  if (!e.second)
  {
    dolfin_error("Parameters.cpp",
                 "add parameter",
                 "Parameter \"%s.%s\" already defined",
                 this->name().c_str(), key.c_str());
  }
}
//-----------------------------------------------------------------------------
void Parameters::add(std::string key, int value,
                     int min_value, int max_value)
{
  auto e = _parameters.insert({key, Parameter(key, value)});
  if (!e.second)
  {
    dolfin_error("Parameters.cpp",
                 "add parameter",
                 "Parameter \"%s.%s\" already defined",
                 this->name().c_str(), key.c_str());
  }

  e.first->second.set_range(min_value, max_value);
}
//-----------------------------------------------------------------------------
void Parameters::add(std::string key, double value)
{
  auto e = _parameters.insert({key, Parameter(key, value)});
  if (!e.second)
  {
    dolfin_error("Parameters.cpp",
                 "add parameter",
                 "Parameter \"%s.%s\" already defined",
                 this->name().c_str(), key.c_str());
  }
}
//-----------------------------------------------------------------------------
void Parameters::add(std::string key, double value,
                     double min_value, double max_value)
{
  auto e = _parameters.insert({key, Parameter(key, value)});
  if (!e.second)
  {
    dolfin_error("Parameters.cpp",
                 "add parameter",
                 "Parameter \"%s.%s\" already defined",
                 this->name().c_str(), key.c_str());
  }

  e.first->second.set_range(min_value, max_value);
}
//-----------------------------------------------------------------------------
void Parameters::add(std::string key, std::string value)
{
  auto e = _parameters.insert({key, Parameter(key, value)});
  if (!e.second)
  {
    dolfin_error("Parameters.cpp",
                 "add parameter",
                 "Parameter \"%s.%s\" already defined",
                 this->name().c_str(), key.c_str());
  }
}
//-----------------------------------------------------------------------------
void Parameters::add(std::string key, const char* value)
{
  // This version is needed to avoid having const char* picked up by
  // the add function for bool parameters.

  auto e = _parameters.insert({key, Parameter(key, value)});
  if (!e.second)
  {
    dolfin_error("Parameters.cpp",
                 "add parameter",
                 "Parameter \"%s.%s\" already defined",
                 this->name().c_str(), key.c_str());
  }
}
//-----------------------------------------------------------------------------
void Parameters::add(std::string key, std::string value,
                     std::set<std::string> range)
{
  auto e = _parameters.insert({key, Parameter(key, value)});
  if (!e.second)
  {
    dolfin_error("Parameters.cpp",
                 "add parameter",
                 "Parameter \"%s.%s\" already defined",
                 this->name().c_str(), key.c_str());
  }

  e.first->second.set_range(range);
}
//-----------------------------------------------------------------------------
void Parameters::add(std::string key, const char* value,
                     std::set<std::string> range)
{
  // This version is needed to avoid having const char* picked up by
  // the add function for bool parameters.

  auto e = _parameters.insert({key, Parameter(key, value)});
  if (!e.second)
  {
    dolfin_error("Parameters.cpp",
                 "add parameter",
                 "Parameter \"%s.%s\" already defined",
                 this->name().c_str(), key.c_str());
  }

  e.first->second.set_range(range);
}
//-----------------------------------------------------------------------------
void Parameters::add(std::string key, bool value)
{
  auto e = _parameters.insert({key, Parameter(key, value)});
  if (!e.second)
  {
    dolfin_error("Parameters.cpp",
                 "add parameter",
                 "Parameter \"%s.%s\" already defined",
                 this->name().c_str(), key.c_str());
  }
}
//-----------------------------------------------------------------------------
void Parameters::add(const Parameters& parameters)
{
  auto e =_parameter_sets.insert({parameters.name(), Parameters(parameters)});
  if (!e.second)
  {
    dolfin_error("Parameters.cpp",
                 "add parameter set",
                 "Parameter set \"%s.%s\" already defined",
                 this->name().c_str(), parameters.name().c_str());
  }

  // Add parameter set

}
//-----------------------------------------------------------------------------
void Parameters::remove(std::string key)
{
  std::size_t num_removed = 0;
  num_removed += _parameters.erase(key);
  num_removed += _parameter_sets.erase(key);

  if (num_removed == 0)
  {
    dolfin_error("Parameters.cpp",
                 "remove parameter or parameter set",
                 "No parameter or parameter set \"%s.%s\" defined",
                 this->name().c_str(), key.c_str());
  }

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
  for (auto it = parameters._parameters.begin(); it != parameters._parameters.end(); ++it)
  {
    // Get parameters
    const Parameter& other = it->second;
    auto this_it = _parameters.find(other.key());

    if (this_it == _parameters.end())
    {
      // This set does not have a parameter that is present in 'other'
      warning("Ignoring unknown parameter \"%s\" in parameter set \"%s\" when updating parameter set \"%s\".",
              other.key().c_str(), parameters.name().c_str(), name().c_str());
    }
    else
    {
      // Skip unset parameters
      if (!other.is_set())
      {
        //warning("Ignoring unset parameter \"%s\" in parameter set \"%s\" when updating parameter set \"%s\".",
        //        other.key().c_str(), parameters.name().c_str(), name().c_str());
      }
      else
      {
        // Update value
        this_it->second = other;
      }
    }
  }

  // Update nested parameter sets
  for (auto it = parameters._parameter_sets.begin(); it != parameters._parameter_sets.end(); ++it)
    (*this)(it->first).update(it->second);
}
//-----------------------------------------------------------------------------
Parameter& Parameters::operator[] (std::string key)
{
  auto p = _parameters.find(key);
  if (p == _parameters.end())
  {
    dolfin_error("Parameters.cpp",
                 "access parameter",
                 "Parameter \"%s.%s\" not defined",
                 this->name().c_str(), key.c_str());
  }

  return p->second;
}
//-----------------------------------------------------------------------------
const Parameter& Parameters::operator[] (std::string key) const
{
  auto p = _parameters.find(key);
  if (p == _parameters.end())
  {
    dolfin_error("Parameters.cpp",
                 "access parameter",
                 "Parameter \"%s.%s\" not defined",
                 this->name().c_str(), key.c_str());
  }

  return p->second;
}
//-----------------------------------------------------------------------------
Parameters& Parameters::operator() (std::string key)
{
  auto p = _parameter_sets.find(key);
  if (p == _parameter_sets.end())
  {
    dolfin_error("Parameters.cpp",
                 "access parameter set",
                 "Parameter set \"%s.%s\" not defined",
                 this->name().c_str(), key.c_str());
  }

  return p->second;
}
//-----------------------------------------------------------------------------
const Parameters& Parameters::operator() (std::string key) const
{
  auto p = _parameter_sets.find(key);
  if (p == _parameter_sets.end())
  {
    dolfin_error("Parameters.cpp",
                 "access parameter set",
                 "Parameter set \"%s.%s\" not defined",
                 this->name().c_str(), key.c_str());
  }

  return p->second;
}
//-----------------------------------------------------------------------------
const Parameters& Parameters::operator= (const Parameters& parameters)
{
  // Clear all parameters
  clear();

  // Copy key
  _key = parameters._key;

  // Copy parameters and parameter sets
  _parameters = parameters._parameters;
  _parameter_sets = parameters._parameter_sets;

  return *this;
}
//-----------------------------------------------------------------------------
bool Parameters::has_key(std::string key) const
{
  return has_parameter(key) or has_parameter_set(key);
}
//-----------------------------------------------------------------------------
bool Parameters::has_parameter(std::string key) const
{
  return _parameters.find(key) != _parameters.end();
}
//-----------------------------------------------------------------------------
bool Parameters::has_parameter_set(std::string key) const
{
  return _parameter_sets.find(key) != _parameter_sets.end();
}
//-----------------------------------------------------------------------------
void Parameters::get_parameter_keys(std::vector<std::string>& keys) const
{
  keys.reserve(_parameters.size());
  for (auto it = _parameters.begin(); it != _parameters.end(); ++it)
    keys.push_back(it->first);
}
//-----------------------------------------------------------------------------
void Parameters::get_parameter_set_keys(std::vector<std::string>& keys) const
{
  keys.reserve(_parameter_sets.size());
  for (auto it = _parameter_sets.begin(); it != _parameter_sets.end(); ++it)
    keys.push_back(it->first);
 }
//-----------------------------------------------------------------------------
std::string Parameters::str(bool verbose) const
{
  std::stringstream s;
  if (verbose)
  {
    s << str(false) << std::endl << std::endl;
    if (_parameters.empty() and _parameter_sets.empty())
    {
      s << name() << indent("(empty)");
      return s.str();
    }

    Table t(_key);
    for (auto it = _parameters.begin(); it != _parameters.end(); ++it)
    {
      const Parameter& p = it->second;
      t(p.key(), "type") = p.type_str();
      t(p.key(), "value") = (p.is_set() ? p.value_str() : "<unset>");
      t(p.key(), "range") = p.range_str();
      t(p.key(), "access") = p.access_count();
      t(p.key(), "change") = p.change_count();
    }
    s << indent(t.str(true));

    for (auto it = _parameter_sets.begin(); it != _parameter_sets.end(); ++it)
      s << "\n\n" << indent(it->second.str(verbose));
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

  // FIXME: This is commented out because it generated a lot of
  // misleading noise with application-specific user parameters,
  // especially when running in parallel

  // Collect and report unrecognized options
  /*
  const std::vector<std::string> unrecognized_options
    = po::collect_unrecognized(parsed.options, po::include_positional);
  for (std::size_t i = 0; i < unrecognized_options.size(); ++i)
  {
    std::cout << "Skipping unrecognized option for parameter set \""
              << name() << "\": " << unrecognized_options[i] << std::endl;
  }
  */

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
  for (auto it = parameters._parameters.begin(); it != parameters._parameters.end(); ++it)
  {
    const Parameter& p = it->second;
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

  for (auto it = parameters._parameter_sets.begin();
       it != parameters._parameter_sets.end(); ++it)
  {
    add_parameter_set_to_po(desc, it->second, base_name + it->first + ".");
  }
}
//-----------------------------------------------------------------------------
void Parameters::read_vm(po::variables_map& vm, Parameters& parameters,
                         std::string base_name)
{
  // Read values from po::variables_map
  for (auto it = parameters._parameters.begin(); it != parameters._parameters.end(); ++it)
  {
    Parameter& p = it->second;
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

  for (auto it = parameters._parameter_sets.begin();
       it != parameters._parameter_sets.end(); ++it)
  {
    read_vm(vm, it->second, base_name + it->first + ".");
  }
}
//-----------------------------------------------------------------------------
boost::optional<Parameter&> Parameters::find_parameter(std::string key)
{
  auto p = _parameters.find(key);
  if (p == _parameters.end())
    return boost::none;
  else
    return p->second;
}
//-----------------------------------------------------------------------------
boost::optional<Parameters&> Parameters::find_parameter_set(std::string key)
{
  auto p = _parameter_sets.find(key);
  if (p == _parameter_sets.end())
    return boost::none;
  else
    return p->second;
}
//-----------------------------------------------------------------------------
namespace dolfin
{
  Parameters empty_parameters("empty");
}
//-----------------------------------------------------------------------------
