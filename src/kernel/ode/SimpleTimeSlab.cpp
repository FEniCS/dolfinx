// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <iostream>

#include <dolfin/dolfin_log.h>
#include <dolfin/Element.h>
#include <dolfin/TimeSlabData.h>
#include <dolfin/Partition.h>
#include <dolfin/RHS.h>
#include <dolfin/SimpleTimeSlab.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
SimpleTimeSlab::SimpleTimeSlab(real t0, real t1, RHS& f, TimeSlabData& data) : 
  TimeSlab(t0, t1)
{
  create(f, data);
}
//-----------------------------------------------------------------------------
SimpleTimeSlab::~SimpleTimeSlab()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void SimpleTimeSlab::update(RHS& f, TimeSlabData& data)
{
  // Update all elements
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    
    // Update element
    element->update(f);
    
    // Write debug info
    data.debug(*element, TimeSlabData::update);
  }
}
//-----------------------------------------------------------------------------
void SimpleTimeSlab::create(RHS& f, TimeSlabData& data)
{
  dolfin_debug("Creating simple time slab");

  // FIXME: choose element and order here
  Element::Type type = Element::cg;
  int q = 1;

  // Get initial time step (same for all components)
  real k = data.regulator(0).timestep();

  // Set size of this time slab
  setsize(k, data);

  // Create elements
  for (unsigned int i = 0; i < data.size(); i++)
  {
    // Create element
    Element *element = data.createElement(type, q, i, this);
    
    // Write debug info
    data.debug(*element, TimeSlabData::create);

    // Add element to array
    elements.push_back(element);
  }
}
//-----------------------------------------------------------------------------
