// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SPECIAL_FUNCTIONS_H
#define __SPECIAL_FUNCTIONS_H

namespace dolfin
{

  /// This function represents the local mesh size on a given mesh.
  class MeshSize : public Function
  {
    real eval(const Point& p, unsigned int i)
    {
      return cell().diameter();
    }
  };

  /// This is the zero function.
  class Zero : public Function
  {
    real eval(const Point& p, unsigned int i)
    {
      return 0.0;
    }
  };

  /// This is the unity function.
  class Unity : public Function
  {
    real eval(const Point& p, unsigned int i)
    {
      return 1.0;
    }
  };

}

#endif
