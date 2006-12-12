// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-05-28
// Last changed: 2006-12-12

#include <dolfin/FiniteElement.h>
#include <dolfin/BilinearForm.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
BilinearForm::BilinearForm(uint num_functions)
  : Form(num_functions), _test(0), _trial(0), test_nodes(0), trial_nodes(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BilinearForm::~BilinearForm()
{
  if ( _test ) delete _test;
  if ( _trial ) delete _trial;
  if ( test_nodes ) delete [] test_nodes;
  if ( trial_nodes ) delete [] trial_nodes;
}
//-----------------------------------------------------------------------------
FiniteElement& BilinearForm::test()
{
  dolfin_assert(_test); // Should be created by child class
  return *_test;
}
//-----------------------------------------------------------------------------
const FiniteElement& BilinearForm::test() const
{
  dolfin_assert(_test); // Should be created by child class
  return *_test;
}
//-----------------------------------------------------------------------------
FiniteElement& BilinearForm::trial()
{
  dolfin_assert(_trial); // Should be created by child class
  return *_trial;
}
//-----------------------------------------------------------------------------
const FiniteElement& BilinearForm::trial() const
{
  dolfin_assert(_trial); // Should be created by child class
  return *_trial;
}
//-----------------------------------------------------------------------------
void BilinearForm::updateLocalData()
{
  // Initialize block
  const uint m = _test->spacedim();
  const uint n = _trial->spacedim();
  if ( !block )
    block = new real[m*n];
  for (uint i = 0; i < m*n; i++)
    block[i] = 0.0;

  // Initialize nodes
  if ( !test_nodes )
    test_nodes = new int[_test->spacedim()];
  if ( !trial_nodes )
    trial_nodes = new int[_trial->spacedim()];
}
//-----------------------------------------------------------------------------
