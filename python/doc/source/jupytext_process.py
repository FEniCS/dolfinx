# Copyright (C) 2017-2022 Garth N. Wells, Jack S. Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import shutil
import pathlib

import jupytext


def process():
    """Convert light format demo Python files into myrst flavoured markdown and
    ipynb using Jupytext. These files can then be included in Sphinx
    documentation"""
    # Directories to scan
    subdirs = [pathlib.Path("../../demo")]

    # Iterate over subdirectories containing demos
    for subdir in subdirs:

        # Build list of demo files
        demos = list(subdir.glob('**/demo*.py'))

        # Make demo doc directory
        demo_dir = pathlib.Path('./demos')
        demo_dir.mkdir(parents=True, exist_ok=True)

        # Process each demo using jupytext/myst
        for demo in demos:
            python_demo = jupytext.read(demo)
            myst_text = jupytext.writes(python_demo, fmt="myst")

            # myst-parser does not process blocks with {code-cell}
            myst_text = myst_text.replace("{code-cell}", "python")
            myst_file = (demo_dir / demo.name).with_suffix(".md")
            with open(myst_file, "w") as fw:
                fw.write(myst_text)

            ipynb_file = (demo_dir / demo.name).with_suffix(".ipynb")
            jupytext.write(python_demo, ipynb_file, fmt="ipynb")

            # Copy python demo files into documentation demo directory
            for f in demos:
                shutil.copy(f, demo_dir)


if __name__ == "__main__":
    process()
