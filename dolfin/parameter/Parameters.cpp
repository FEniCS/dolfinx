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
typedef std::map<std::string, NewParameter*>::iterator iterator;
typedef std::map<std::string, NewParameter*>::const_iterator const_iterator;

//-----------------------------------------------------------------------------
Parameters::Parameters(std::string name) : _name(name)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Parameters::~Parameters()
{
  // Delete parameters
  for (iterator it = _parameters.begin(); it != _parameters.end(); ++it)
    delete it->second;
}
//-----------------------------------------------------------------------------
std::string Parameters::name() const
{
  return _name;
}
//-----------------------------------------------------------------------------
void Parameters::add(std::string key, int value)
{
  // Check key name
  if (find(key))
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
  NewParameter* p = find(key);
  dolfin_assert(p);
  p->set_range(min_value, max_value);
}
//-----------------------------------------------------------------------------
void Parameters::add(std::string key, double value)
{
  // Check key name
  if (find(key))
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
  NewParameter* p = find(key);
  dolfin_assert(p);
  p->set_range(min_value, max_value);
}
//-----------------------------------------------------------------------------
NewParameter& Parameters::operator[] (std::string key)
{
  NewParameter* p = find(key);
  if (!p)
    error("Unable to access parameter \"%s\", parameter not defined.",
          key.c_str());
  return *p;
}
//-----------------------------------------------------------------------------
const NewParameter& Parameters::operator[] (std::string key) const
{
  NewParameter* p = find(key);
  if (!p)
    error("Unable to access parameter \"%s\", parameter not defined.",
          key.c_str());
  return *p;
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
  Table t(_name);
  for (const_iterator it = _parameters.begin(); it != _parameters.end(); ++it)
  {
    NewParameter* p = it->second;
    t(p->key(), "value type") = p->type_str();
    t(p->key(), "value") = p->value_str();
    t(p->key(), "range") = p->range_str();
    t(p->key(), "access count") = p->access_count();
    t(p->key(), "change count") = p->change_count();
  }

  t.print();
}
//-----------------------------------------------------------------------------
NewParameter* Parameters::find(std::string key) const
{
  const_iterator p = _parameters.find(key);
  if (p == _parameters.end())
    return 0;
  return p->second;
}
//-----------------------------------------------------------------------------
