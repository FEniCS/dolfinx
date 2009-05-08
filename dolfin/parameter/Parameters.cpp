// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-05-08
// Last changed: 2009-05-08

#include <sstream>
#include <dolfin/log/log.h>
#include <dolfin/log/Table.h>
#include "Parameters.h"

using namespace dolfin;

// Typedef of iterators for convenience
typedef std::map<std::string, NewParameter*>::iterator parameter_iterator;
typedef std::map<std::string, NewParameter*>::const_iterator const_parameter_iterator;
typedef std::map<std::string, Parameters*>::iterator database_iterator;
typedef std::map<std::string, Parameters*>::const_iterator const_database_iterator;

//-----------------------------------------------------------------------------
Parameters::Parameters(std::string key) : _key(key)
{
  // Do nothing
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
std::string Parameters::key() const
{
  return _key;
}
//-----------------------------------------------------------------------------
void Parameters::clear()
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
void Parameters::add(std::string key, int value)
{
  // Check key name
  if (find_parameter(key))
    error("Unable to add parameter \"%s\", already defined.", key.c_str());

  // Add parameter
  _parameters[key] = new NewIntParameter(key, value);
}
//-----------------------------------------------------------------------------
void Parameters::add(std::string key, int value,
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
void Parameters::add(std::string key, double value)
{
  // Check key name
  if (find_parameter(key))
    error("Unable to add parameter \"%s\", already defined.", key.c_str());

  // Add parameter
  _parameters[key] = new NewDoubleParameter(key, value);
}
//-----------------------------------------------------------------------------
void Parameters::add(std::string key, double value,
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
void Parameters::add(const Parameters& parameters)
{
  // Check key name
  if (find_database(parameters.key()))
    error("Unable to add parameter database \"%s\", already defined.",
          parameters.key().c_str());

  // Add parameter database
  Parameters* p = new Parameters("");
  *p = parameters;
  _databases[parameters.key()] = p;
}
//-----------------------------------------------------------------------------
NewParameter& Parameters::operator() (std::string key)
{
  NewParameter* p = find_parameter(key);
  if (!p)
    error("Unable to access parameter \"%s\", parameter not defined.",
          key.c_str());
  return *p;
}
//-----------------------------------------------------------------------------
const NewParameter& Parameters::operator() (std::string key) const
{
  NewParameter* p = find_parameter(key);
  if (!p)
    error("Unable to access parameter \"%s\", parameter not defined.",
          key.c_str());
  return *p;
}
//-----------------------------------------------------------------------------
Parameters& Parameters::operator[] (std::string key)
{
  Parameters* p = find_database(key);
  if (!p)
    error("Unable to access parameter database \"%s\", database not defined.",
          key.c_str());
  return *p;
}
//-----------------------------------------------------------------------------
const Parameters& Parameters::operator[] (std::string key) const
{
  Parameters* p = find_database(key);
  if (!p)
    error("Unable to access parameter database \"%s\", database not defined.",
          key.c_str());
  return *p;
}
//-----------------------------------------------------------------------------
const Parameters& Parameters::operator= (const Parameters& parameters)
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
    else
      error("Unable to copy parameter, unknown type: \"%s\".",
            p.type_str().c_str());
    _parameters[p.key()] = q;
  }

  // Copy databases
  for (const_database_iterator it = parameters._databases.begin();
       it != parameters._databases.end(); ++it)
  {
    const Parameters& p = *it->second;
    _databases[p.key()] = new Parameters(p);
  }

  return *this;
}
//-----------------------------------------------------------------------------
std::string Parameters::str() const
{
  std::stringstream s;
  s << "<Parameter database containing " << _parameters.size() << " parameters>";
  return s.str();
}
//-----------------------------------------------------------------------------
void Parameters::print() const
{
  if (_parameters.size() > 0)
  {
    info_underline("Parameters");
    info("");
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
  t.print();

  if (_databases.size() > 0)
  {
    info("");
    info_underline("Nested parameter databases");
  }

  begin(" ");
  for (const_database_iterator it = _databases.begin(); it != _databases.end(); ++it)
    it->second->print();
  end();
}
//-----------------------------------------------------------------------------
NewParameter* Parameters::find_parameter(std::string key) const
{
  const_parameter_iterator p = _parameters.find(key);
  if (p == _parameters.end())
    return 0;
  return p->second;
}
//-----------------------------------------------------------------------------
Parameters* Parameters::find_database(std::string key) const
{
  const_database_iterator p = _databases.find(key);
  if (p == _databases.end())
    return 0;
  return p->second;
}
//-----------------------------------------------------------------------------
