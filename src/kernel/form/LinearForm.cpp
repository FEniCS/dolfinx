// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-05-28
// Last changed: 2006-12-07

#include <dolfin/LinearForm.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearForm::LinearForm(uint num_functions)
  : Form(num_functions), _test(0), test_nodes(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
LinearForm::~LinearForm()
{
  if ( _test ) delete _test;
  if ( test_nodes ) delete [] test_nodes;
}
//-----------------------------------------------------------------------------
void LinearForm::update(AffineMap& map)
{
  // Update coefficients
  updateCoefficients(map);

  // Update local data structures
  updateLocalData();
}
//-----------------------------------------------------------------------------
FiniteElement& LinearForm::test()
{
  dolfin_assert(_test); // Should be created by child class
  return *_test;
}
//-----------------------------------------------------------------------------
const FiniteElement& LinearForm::test() const
{
  dolfin_assert(_test); // Should be created by child class
  return *_test;
}
//-----------------------------------------------------------------------------
void LinearForm::updateLocalData()
{
  // Initialize block
  const uint m = _test->spacedim();
  if ( !block )
    block = new real[m];
  for (uint i = 0; i < m; i++)
    block[i] = 0.0;

  // Initialize nodes
  if ( !test_nodes )
    test_nodes = new int[_test->spacedim()];
}
//-----------------------------------------------------------------------------
