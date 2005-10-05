// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-10-03
// Last changed: 2005-10-03

#ifndef __BLAS_FORM_DATA_H
#define __BLAS_FORM_DATA_H

#include <dolfin/constants.h>
#include <dolfin/Array.h>

namespace dolfin
{
 
  /// BLAS form data for FFC BLAS mode.
  
  class BLASFormData
  {
  public:

    /// Constructor
    BLASFormData();

    /// Destructor
    ~BLASFormData();

    /// Initialize from given form data file
    void init(const char* filename);

    /// Initialize from given data
    void init(uint mi, uint ni, const Array<Array<real> > data_interior,
	      uint mb, uint nb, const Array<Array<real> > data_boundary);

    /// Clear data
    void clear();

    /// Display data
    void disp() const;

    /// Data arrays
    real* Ai;
    real* Gi;
    real* Ab;
    real* Gb;
    
    /// Matrix dimensions
    uint mi, ni;
    uint mb, nb;

  private:

    // Initialize data (independent of interiori/boundary)
    void init(uint m, uint n, const Array<Array<real> >& data, real** A, real** G);

  };

}

#endif
