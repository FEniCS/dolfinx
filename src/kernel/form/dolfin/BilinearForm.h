// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-05-28
// Last changed: 2006-12-06

#ifndef __BILINEAR_FORM_H
#define __BILINEAR_FORM_H

#include <dolfin/Form.h>

namespace dolfin
{

  /// BilinearForm represents a multilinear form of the type
  ///
  ///     a = a(v1, v2, w1, w2, ..., wn)
  ///
  /// where the first two arguments v1 and v2 are basis functions
  /// (the test and trial functions) and where w1, w2, ..., wn are
  /// any given functions.

  class BilinearForm : public Form
  {
  public:
    
    /// Constructor
    BilinearForm(uint num_functions = 0);
    
    /// Destructor
    virtual ~BilinearForm();

    /// Check if there is a contribution from the interior
    virtual bool interior_contribution() const = 0;
    
    /// Compute element matrix (interior contribution)
    virtual void eval(real block[], const AffineMap& map, real det) const = 0;
    
    /// Check if there is a contribution from the boundary
    virtual bool boundary_contribution() const = 0;

    /// Compute element matrix (boundary contribution)
    virtual void eval(real block[], const AffineMap& map, real det, uint segment) const = 0;

    /// Check if there is a contribution from the interior boundary
    virtual bool interior_boundary_contribution() const = 0;

    /// Compute exterior facet tensor
    virtual void eval(real block[],
                      const AffineMap& map0, const AffineMap& map1, real det,
                      uint facet0, uint facet1, uint alignment) const = 0;

    /// Update map to current cell
    void update(AffineMap& map);

    /// Update map to current pair of cells for interior facet
    void update(AffineMap& map0, AffineMap& map1);

    /// Return finite element defining the test space
    FiniteElement& test();

    /// Return finite element defining the test space
    const FiniteElement& test() const;

    /// Return finite element defining the trial space
    FiniteElement& trial();

    /// Return finite element defining the trial space
    const FiniteElement& trial() const;

    /// Friends
    friend class FEM;

  protected:

    // Finite element defining the test space
    FiniteElement* _test;

    // Finite element defining the trial space
    FiniteElement* _trial;

    // Local-to-global mapping for test space
    int* test_nodes;

    // Local-to-global mapping for trial space
    int* trial_nodes;

  };

}

#endif
