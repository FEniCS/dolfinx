// Copyright (C) 2009 Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added: 2009-03-17
// Last changed: 2009-03-17

#include "XMLIndent.h"
#include <iomanip>

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLIndent::XMLIndent(uint indentation_level, uint step_size)
  : indentation_level(indentation_level), step_size(step_size)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLIndent::~XMLIndent()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLIndent::operator++()
{
  ++indentation_level;
}
//-----------------------------------------------------------------------------
void XMLIndent::operator--()
{
  -- indentation_level;
}
//-----------------------------------------------------------------------------
std::string XMLIndent::operator()()
{
  std::ostringstream ss;
  ss << std::setw(indentation_level*step_size) << "";
  return ss.str();
}
//-----------------------------------------------------------------------------
