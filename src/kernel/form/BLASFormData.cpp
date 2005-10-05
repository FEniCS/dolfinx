// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-10-03
// Last changed: 2005-10-03

#include <dolfin/dolfin_log.h>
#include <dolfin/File.h>
#include <dolfin/BLASFormData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
BLASFormData::BLASFormData()
  : Ai(0), Gi(0), Ab(0), Gb(0), mi(0), ni(0), mb(0), nb(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BLASFormData::~BLASFormData()
{
  clear();
}
//-----------------------------------------------------------------------------
void BLASFormData::init(const char* filename)
{
  // Read data from file (will call init() function below with data)
  File file(filename);
  file >> *this;
}
//-----------------------------------------------------------------------------
void BLASFormData::init(uint mi, uint ni, 
			const Array<Array<real> > data_interior,
			uint mb, uint nb,
			const Array<Array<real> > data_boundary)
{
  // Compute total size of data
  uint size_interior = 0;
  for (uint i = 0; i < data_interior.size(); i++)
    size_interior += data_interior[i].size();
  uint size_boundary = 0;
  for (uint i = 0; i < data_boundary.size(); i++)
    size_boundary += data_boundary[i].size();
  
  // Check data dimensions
  if ( mi * ni != size_interior )
    dolfin_error("Inconsistent BLAS form data for interior contribution.");
  if ( mb * nb != size_boundary )
    dolfin_error("Inconsistent BLAS form data for boundary contribution.");

  // Clear old data if any
  clear();

  // Allocate and initialize arrays
  init(mi, ni, data_interior, &Ai, &Gi);
  init(mb, nb, data_boundary, &Ab, &Gb);

  // Remember size
  this->mi = mi;
  this->ni = ni;
  this->mb = mb;
  this->nb = nb;
}
//-----------------------------------------------------------------------------
void BLASFormData::clear()
{
  if ( Ai )
    delete [] Ai;
  Ai = 0;

  if ( Gi )
    delete [] Gi;
  Gi = 0;

  if ( Ab )
    delete [] Ab;
  Ab = 0;

  if ( Gb )
    delete [] Gb;
  Gb = 0;
}
//-----------------------------------------------------------------------------
void BLASFormData::disp() const
{
  dolfin_info("BLAS form data, interior contribution (m = %d, n = %d):", mi, ni);
  for (uint i = 0; i < mi; i++)
  {
    for (uint j = 0; j < ni; j++)
    {
      cout << Ai[i*ni + j];
      if ( j < (ni - 1) )
	cout << " ";
      else
	cout << endl;
    }
  }

  cout << endl;

  dolfin_info("BLAS form data, boundary contribution (m = %d, n = %d):", mb, nb);
  for (uint i = 0; i < mb; i++)
  {
    for (uint j = 0; j < nb; j++)
    {
      cout << Ab[i*nb + j];
      if ( j < (nb - 1) )
	cout << " ";
      else
	cout << endl;
    }
  }
}
//-----------------------------------------------------------------------------
void BLASFormData::init(uint m, uint n, const Array<Array<real> >& data,
			real** A, real** G)
{
  if ( m*n == 0 )
    return;

  // We get an array of arrays, representing an array of matrices
  // (which in turn represent an array of tensors). Instead of doing
  // the matrix-vector multiplications term by term, we stack the
  // matrices next to each other. To complicate things even more, the
  // new big matrix is stored as one big array to make BLAS happy.

  cout << "Allocating, size = " << m*n << endl;

  (*A) = new real[m*n];
  uint offset_cols = 0;
  for (uint term = 0; term < data.size(); term++)
  {
    const Array<real>& tensor = data[term];
    const uint cols = tensor.size() / m;
    for (uint pos = 0; pos < tensor.size(); pos++)
    {
      const uint i = pos / cols;
      const uint j = pos % cols;
      (*A)[i*n + offset_cols + j] = tensor[pos];
    }
    offset_cols += cols;
  }

  (*G) = new real[n];
  for (uint i = 0; i < n; i++)
    (*G)[i] = 0.0;
}
//-----------------------------------------------------------------------------

