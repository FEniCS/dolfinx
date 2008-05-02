// Copyright (C) 2008 Solveig Bruvoll and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-02
// Last changed: 2008-05-02

#ifndef __ALE_METHOD_H
#define __ALE_METHOD_H

namespace dolfin
{

  /// List of available methods for ALE mesh movement
  enum ALEMethod {lagrange, hermite, harmonic, elastic};

}

#endif
