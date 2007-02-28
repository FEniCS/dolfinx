// Copyright (C) 2007 Anders Logg and ...
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-01-17
// Last changed: 2007-01-17

#include <dolfin/dolfin_log.h>
#include <dolfin/GenericTensor.h>
#include <dolfin/Mesh.h>
#include <dolfin/Cell.h>
#include <dolfin/AssemblyData.h>
#include <dolfin/Assembler.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Assembler::Assembler()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Assembler::~Assembler()
{


}
//-----------------------------------------------------------------------------
void Assembler::assemble(GenericTensor& A, const ufc::form& form, Mesh& mesh)
{
  cout << "Assembling form " << form.signature() << endl;

  // Create data structure for local assembly data
  AssemblyData data(form);

  // Update dof map storage for current form
  dof_maps.update(form, mesh);
  
  // Iterate over the cells of the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    cout << *cell << endl;
    
    
  }
}
//-----------------------------------------------------------------------------
