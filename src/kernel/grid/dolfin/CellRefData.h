// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __CELL_REF_DATA_H
#define __CELL_REF_DATA_H

namespace dolfin {

  enum CellMarker { marked_for_reg_ref,        // Marked for regular refinement
		    marked_for_irr_ref_1,      // Marked for irregular refinement by rule 1
		    marked_for_irr_ref_2,      // Marked for irregular refinement by rule 2
		    marked_for_irr_ref_3,      // Marked for irregular refinement by rule 3
		    marked_for_irr_ref_4,      // Marked for irregular refinement by rule 4
		    marked_for_no_ref,         // Marked for no refinement
		    marked_for_coarsening,     // Marked for coarsening
		    marked_according_to_ref }; // Marked according to refinement
  
  enum CellStatus { ref_reg,                   // Refined regularly
		    ref_irr,                   // Refined irregularly
		    unref };                   // Unrefined

  /// Cell refinement data
  class CellRefData {
  public:

    /// Create cell refinement data
    CellRefData() {
      marker = marked_according_to_ref;
      status = unref;
      closed = false;
    }
    
    /// The mark of the cell
    CellMarker marker;

    /// The status of the cell
    CellStatus status;

    /// True if cell has been closed
    bool closed;
    
    /// True if the cell is marked for re-use
    bool re_use;

  };

}

#endif
