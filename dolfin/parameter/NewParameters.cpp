// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Hake, 2009
//
// First added:  2009-05-08
// Last changed: 2009-05-23

#include <sstream>
#include <boost/program_options.hpp>

#include <dolfin/log/log.h>
#include <dolfin/log/Table.h>
#include "NewParameter.h"
#include "NewParameters.h"

using namespace dolfin;
namespace po = boost::program_options;

// Typedef of iterators for convenience
typedef std::map<std::string, NewParameter*>::iterator parameter_iterator;
typedef std::map<std::string, NewParameter*>::const_iterator const_parameter_iterator;
typedef std::map<std::string, NewParameters*>::iterator database_iterator;
typedef std::map<std::string, NewParameters*>::const_iterator const_database_iterator;

//-----------------------------------------------------------------------------
NewParameters::NewParameters(std::string key) : _key(key)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewParameters::~NewParameters()
{
  clear();
}
//-----------------------------------------------------------------------------
NewParameters::NewParameters(const NewParameters& parameters)
{
  *this = parameters;
}
//-----------------------------------------------------------------------------
std::string NewParameters::key() const
{
  return _key;
}
//-----------------------------------------------------------------------------
void NewParameters::clear()
{
  // Delete parameters
  for (parameter_iterator it = _parameters.begin(); it != _parameters.end(); ++it)
    delete it->second;
  _parameters.clear();

  // Delete database
  for (database_iterator it = _databases.begin(); it != _databases.end(); ++it)
    delete it->second;
  _databases.clear();

  // Reset key
  _key = "";
}
//-----------------------------------------------------------------------------
void NewParameters::add(std::string key, int value)
{
  // Check key name
  if (find_parameter(key))
    error("Unable to add parameter \"%s\", already defined.", key.c_str());

  // Add parameter
  _parameters[key] = new NewIntParameter(key, value);
}
//-----------------------------------------------------------------------------
void NewParameters::add(std::string key, int value,
                        int min_value, int max_value)
{
  // Add parameter
  add(key, value);

  // Set range
  NewParameter* p = find_parameter(key);
  dolfin_assert(p);
  p->set_range(min_value, max_value);
}
//-----------------------------------------------------------------------------
void NewParameters::add(std::string key, double value)
{
  // Check key name
  if (find_parameter(key))
    error("Unable to add parameter \"%s\", already defined.", key.c_str());

  // Add parameter
  _parameters[key] = new NewDoubleParameter(key, value);
}
//-----------------------------------------------------------------------------
void NewParameters::add(std::string key, double value,
                        double min_value, double max_value)
{
  // Add parameter
  add(key, value);

  // Set range
  NewParameter* p = find_parameter(key);
  dolfin_assert(p);
  p->set_range(min_value, max_value);
}
//-----------------------------------------------------------------------------
void NewParameters::add(std::string key, std::string value)
{
  // Check key name
  if (find_parameter(key))
    error("Unable to add parameter \"%s\", already defined.", key.c_str());

  // Add parameter
  _parameters[key] = new NewStringParameter(key, value);
}
//-----------------------------------------------------------------------------
void NewParameters::add(std::string key, std::string value, std::set<std::string> range)
{
  // Add parameter
  add(key, value);

  // Set range
  NewParameter* p = find_parameter(key);
  dolfin_assert(p);
  p->set_range(range);
}
//-----------------------------------------------------------------------------
void NewParameters::add(const NewParameters& parameters)
{
  // Check key name
  if (find_database(parameters.key()))
    error("Unable to add parameter database \"%s\", already defined.",
          parameters.key().c_str());

  // Add parameter database
  NewParameters* p = new NewParameters("");
  *p = parameters;
  _databases[parameters.key()] = p;
}
//-----------------------------------------------------------------------------
void NewParameters::parse(int argc, char* argv[])
{
  info("Parsing command-line arguments...");

  // Add list of allowed options to po::options_description
  po::options_description desc("Allowed options");
  add_database_to_po(desc, *this);
  
  // Add help option
  desc.add_options()("help", "show help text");

  // Read command-line arguments into po::variables_map
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // FIXME: Should we exit after printing help text?

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
void NewParameters::update(const NewParameters& parameters)
{
  // Update the parameters
  for (const_parameter_iterator it = parameters._parameters.begin();
       it != parameters._parameters.end(); ++it)
  {
    const NewParameter& others = *it->second;
    NewParameter& mine = (*this)(others.key());
    if (others.type_str() == "int")
      mine = int(others);
    else if (others.type_str() == "double")
      mine = double(others);
    else if (others.type_str() == "string")
      mine = std::string(others);
    else
      error("Unable to use parameter \"%s\", unknown type: \"%s\".",
            others.key().c_str(), others.type_str().c_str());
  }
  
  // Update the parameter database
  for (const_database_iterator it = parameters._databases.begin(); it != parameters._databases.end(); ++it)
  {
    (*this)[it->first].update(*it->second);
  }
}
//-----------------------------------------------------------------------------
NewParameter& NewParameters::operator() (std::string key)
{
  NewParameter* p = find_parameter(key);
  if (!p)
    error("Unable to access parameter \"%s\", parameter not defined.",
          key.c_str());
  return *p;
}
//-----------------------------------------------------------------------------
const NewParameter& NewParameters::operator() (std::string key) const
{
  NewParameter* p = find_parameter(key);
  if (!p)
    error("Unable to access parameter \"%s\", parameter not defined.",
          key.c_str());
  return *p;
}
//-----------------------------------------------------------------------------
NewParameters& NewParameters::operator[] (std::string key)
{
  NewParameters* p = find_database(key);
  if (!p)
    error("Unable to access parameter database \"%s\", database not defined.",
          key.c_str());
  return *p;
}
//-----------------------------------------------------------------------------
const NewParameters& NewParameters::operator[] (std::string key) const
{
  NewParameters* p = find_database(key);
  if (!p)
    error("Unable to access parameter database \"%s\", database not defined.",
          key.c_str());
  return *p;
}
//-----------------------------------------------------------------------------
const NewParameters& NewParameters::operator= (const NewParameters& parameters)
{
  // Clear database
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
    const NewParameter& p = *it->second;
    NewParameter* q = 0;
    if (p.type_str() == "int")
      q = new NewIntParameter(dynamic_cast<const NewIntParameter&>(p));
    else if (p.type_str() == "double")
      q = new NewDoubleParameter(dynamic_cast<const NewDoubleParameter&>(p));
    else if (p.type_str() == "string")
      q = new NewStringParameter(dynamic_cast<const NewStringParameter&>(p));
    else
      error("Unable to copy parameter, unknown type: \"%s\".",
            p.type_str().c_str());
    _parameters[p.key()] = q;
  }

  // Copy databases
  for (const_database_iterator it = parameters._databases.begin();
       it != parameters._databases.end(); ++it)
  {
    const NewParameters& p = *it->second;
    _databases[p.key()] = new NewParameters(p);
  }

  return *this;
}
//-----------------------------------------------------------------------------
std::string NewParameters::str() const
{
  std::stringstream s;

  if (_parameters.size() == 0 && _databases.size() == 0)
  {
    s << key() << " (empty)";
    return s.str();
  }

  Table t(_key);
  for (const_parameter_iterator it = _parameters.begin(); it != _parameters.end(); ++it)
  {
    NewParameter* p = it->second;
    t(p->key(), "type") = p->type_str();
    t(p->key(), "value") = p->value_str();
    t(p->key(), "range") = p->range_str();
    t(p->key(), "access") = p->access_count();
    t(p->key(), "change") = p->change_count();
  }
  s << t.str();

  if (_databases.size() > 0)
    s << "\n";

  for (const_database_iterator it = _databases.begin(); it != _databases.end(); ++it)
    s << "\n" << indent(it->second->str());

  return s.str();
}
//-----------------------------------------------------------------------------
void NewParameters::parameter_keys(std::vector<std::string>& keys) const
{
  keys.reserve(_parameters.size());
  for (const_parameter_iterator it = _parameters.begin(); it != _parameters.end(); ++it)
    keys.push_back(it->first);
}
//-----------------------------------------------------------------------------
void NewParameters::database_keys(std::vector<std::string>& keys) const
{
  keys.reserve(_databases.size());
  for (const_database_iterator it = _databases.begin(); it != _databases.end(); ++it)
    keys.push_back(it->first);
 }
//-----------------------------------------------------------------------------
void NewParameters::add_database_to_po(po::options_description& desc, const NewParameters &parameters, std::string base_name) const
{
  for (const_parameter_iterator it = parameters._parameters.begin();
       it != parameters._parameters.end(); ++it)
  {
    const NewParameter& p = *it->second;
    std::string param_name(base_name + p.key());
    if (p.type_str() == "int")
      desc.add_options()(param_name.c_str(), po::value<int>(), p.description().c_str());
    else if (p.type_str() == "double")
      desc.add_options()(param_name.c_str(), po::value<double>(), p.description().c_str());
    else if (p.type_str() == "string")
      desc.add_options()(param_name.c_str(), po::value<std::string>(), p.description().c_str());
  }
  
  for (const_database_iterator it = parameters._databases.begin(); it != parameters._databases.end(); ++it)
  {
    add_database_to_po(desc, *it->second, base_name + it->first + ".");
  }
}
//-----------------------------------------------------------------------------
void NewParameters::read_vm(po::variables_map& vm, NewParameters &parameters, std::string base_name)
{
  // Read values from po::variables_map
  for (parameter_iterator it = parameters._parameters.begin();
       it != parameters._parameters.end(); ++it)
  {
    NewParameter& p = *it->second;
    std::string param_name(base_name + p.key());
    if (p.type_str() == "int")
    {
      const po::variable_value& v = vm[param_name];
      if (!v.empty())
        p = v.as<int>();
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
  for (database_iterator it = parameters._databases.begin(); it != parameters._databases.end(); ++it)
  {
    read_vm(vm, *it->second, base_name + it->first + ".");
  }
  
}
//-----------------------------------------------------------------------------
NewParameter* NewParameters::find_parameter(std::string key) const
{
  const_parameter_iterator p = _parameters.find(key);
  if (p == _parameters.end())
    return 0;
  return p->second;
}
//-----------------------------------------------------------------------------
NewParameters* NewParameters::find_database(std::string key) const
{
  const_database_iterator p = _databases.find(key);
  if (p == _databases.end())
    return 0;
  return p->second;
}
//-----------------------------------------------------------------------------
