// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-09-11
// Last changed: 2008-09-11

#ifndef __FINITE_ELEMENT_H
#define __FINITE_ELEMENT_H

#include <ufc.h>

namespace dolfin
{

  /// This is a wrapper for a UFC finite element (ufc::finite_element).

  class FiniteElement : public ufc::finite_element {};

}
