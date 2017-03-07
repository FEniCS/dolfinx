# Copyright (C) 2005-2010 Anders Logg
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Recompile all ffc forms (use when FFC has been updated)
# This script should be run from the top level directory.

import os
import ffc

# Forms that need special options
use_error = ["AdaptivePoisson.ufl",
             "AdaptiveNavierStokes.ufl",
]

use_representation = {"HyperElasticity.ufl": "uflacs",
                      "CahnHilliard2D.ufl": "uflacs",
                      "CahnHilliard3D.ufl": "uflacs",
}

use_split = ["Poisson2D_1.ufl",
             "Poisson2D_2.ufl",
             "Poisson2D_3.ufl",
             "Poisson2D_4.ufl",
             "Poisson2D_5.ufl",
             "Poisson3D_1.ufl",
             "Poisson3D_2.ufl",
             "Poisson3D_3.ufl",
             "Poisson3D_4.ufl",
             "Poisson3D_5.ufl",
             "CahnHilliard2D.ufl",
             "CahnHilliard3D.ufl",
]

# Forms for which we don't want to generate functions for evaluating
# the basis
skip_basis = ["Poisson2D_5.ufl", "Poisson3D_4.ufl"]

# Directories to scan
subdirs = ["demo", "bench", "test"]

# Compile all form files
topdir = os.getcwd()
for subdir in subdirs:
    for root, dirs, files in os.walk(subdir):
        # Check for .ufl files
        formfiles = [f for f in files if f[-4:] == ".ufl"]
        if not formfiles:
            continue

        # Compile files
        os.chdir(root)
        print("Compiling %d forms in %s..." % (len(formfiles), root))
        for f in formfiles:
            args = ["-l", "dolfin"]
            args.append("-s")
            args.append("-O")

            args.append("-r")
            args.append(use_representation.get(f, "auto"))

            if f in use_split:
                args.append("-fsplit")

            if f in use_error:
                args.append("-e")

            if f in skip_basis:
                args.append("-fno-evaluate_basis")
                args.append("-fno-evaluate_basis_derivatives")

            args.append(f)

            command = "ffc " + " ".join(args) # + " >> compile.log"
            print(command)

            ret = ffc.main(args)
            if ret != 0:
                raise RuntimeError("Unable to compile form: %s/%s" % (root, f))
        os.chdir(topdir)
