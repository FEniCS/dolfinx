# Copyright (C) 2017 Garth N. Wells, Jack S. Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os
import shutil

import jupytext


def process():
    """Convert light format demo Python files into myrst flavoured markdown and
    ipynb using Jupytext. These files can then be included in Sphinx
    documentation"""
    # Directories to scan
    subdirs = ["../../demo"]

    # Iterate over subdirectories containing demos
    for subdir in subdirs:

        # Get list of demos (demo name , subdirectory)
        demos = [(dI, os.path.join(subdir, dI)) for dI in os.listdir(subdir) if os.path.isdir(os.path.join(subdir, dI))]

        # Iterate over demos
        for demo, path in demos:

            # Make demo doc directory
            demo_dir = os.path.join('./demos', demo)
            if not os.path.exists(demo_dir):
                os.makedirs(demo_dir)

            # Build list of py and png files in demo source directory
            py_files = [f for f in os.listdir(path) if os.path.splitext(f)[1] == ".py"]
            other_files = [f for f in os.listdir(path) if os.path.splitext(f)[1] in (".png")]

            # Create directory in documentation tree for demo
            demo_dir = os.path.join('./demos/', demo)
            if not os.path.exists(demo_dir):
                os.makedirs(demo_dir)

            # Copy python files into documentation demo directory
            for f in py_files:
                shutil.copy(os.path.join(path, f), demo_dir)

            # Copy .png files into documentation demo directory
            for f in other_files:
                shutil.copy(os.path.join(path, f), demo_dir)

            # Convert light format python to myst flavoured markdown and ipynb
            for f in py_files:
                py_file = os.path.join(demo_dir, f)
                python_demo = jupytext.read(py_file)

                myst_file = os.path.join(demo_dir, os.path.splitext(f)[0] + ".md")
                myst_text = jupytext.writes(python_demo, fmt="myst")
                # myst-parser does not process blocks with {code-cell}
                myst_text = myst_text.replace("{code-cell}", "")
                with open(myst_file, "w") as fw:
                    fw.write(myst_text)

                ipynb_file = os.path.join(demo_dir, os.path.splitext(f)[0] + ".ipynb")
                jupytext.write(python_demo, ipynb_file, fmt="ipynb")


if __name__ == "__main__":
    process()
