// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __CELL_MARKER_H
#define __CELL_MARKER_H

namespace dolfin {

  enum CellMark   { marked_for_reg_ref,        // Marked for regular refinement
		    marked_for_irr_ref,        // Marked for irregular refinement
		    marked_for_irr_ref_by_1,   // Marked for irregular refinement by 1
		    marked_for_irr_ref_by_2,   // Marked for irregular refinement by 2
		    marked_for_irr_ref_by_3,   // Marked for irregular refinement by 3
		    marked_for_irr_ref_by_4,   // Marked for irregular refinement by 4
		    marked_for_no_ref,         // Marked for no refinement
		    marked_for_coarsening,     // Marked for coarsening
		    marked_according_to_ref }; // Marked according to refinement
  
  enum CellStatus { ref_reg,                   // Refined regularly
		    ref_irr,                   // Refined irregularly
		    ref_irr_by_1,              // Refined irregularly by 1
		    ref_irr_by_2,              // Refined irregularly by 2
		    ref_irr_by_3,              // Refined irregularly by 3
		    ref_irr_by_4,              // Refined irregularly by 4
		    unref };                   // Unrefined

  /// Cell marker
  class CellMarker {
  public:

    /// Create an empty marker
    CellMarker() {
      mark = marked_according_to_ref;
      status = unref;
    }
    
    /// The mark of the cell
    CellMark mark;

    /// The status of the cell
    CellStatus status;
    
  };

}

#endif
