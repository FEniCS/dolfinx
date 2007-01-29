// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-09
// Last changed: 2006-12-07

#ifndef __SPECIAL_FUNCTIONS_H
#define __SPECIAL_FUNCTIONS_H

namespace dolfin
{

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

  /// This function represents the local mesh size on a given mesh.
  class MeshSize : public Function
  {
    real eval(const Point& p, unsigned int i)
    {
      return cell().diameter();
    }
  };

  /// This function represents the inverse of the local mesh size on a given mesh.
  class InvMeshSize : public Function
  {
    real eval(const Point& p, unsigned int i)
    {
      return 1.0/cell().diameter();
    }
  };

  /// This function represents the outward unit normal on mesh facets.
  class FacetNormal : public Function
  {
    real eval(const Point& p, unsigned int i)
    {
      cout << "    Evaluating facet normal at p = " << p << endl;
      cout << "      cell      = " << cell().index() << endl;
      cout << "      facet     = " << facet() << endl;
      cout << "      component = " << i << endl;
      cout << "      value     = " << cell().normal(facet(), i) << endl;
      return cell().normal(facet(), i);
    }
  };


  /// Temporary manually implemented common basis functions
  class ScalarLagrange : public Function
  {
  public:
    ScalarLagrange(int num) : num(num) {}
    
    real eval(const Point& p, unsigned int i)
    {
      switch(num)
      {
      case 0:
	return 1 - p.x() - p.y() - p.z();
	break;
      case 1:
	return p.x();
	break;
      case 2:
	return p.y();
	break;
      case 3:
	return p.z();
	break;
      default:
	return 0.0;
      }
    }
    
  private:
    int num;
  };

}

#endif
