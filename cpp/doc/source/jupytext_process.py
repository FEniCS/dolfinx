# Copyright (C) 2017-2023 Garth N. Wells, Jack S. Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os
import pathlib
import runpy
import shutil

import jupytext


def process():
    """Convert Python demos in the Jupytext 'light' format into MyST
    flavoured markdown and ipynb using Jupytext. These files can then be
    included in Sphinx documentation.

    """
    # Make demo doc directory
    demo_doc_dir = pathlib.Path("./demos")
    demo_doc_dir.mkdir(parents=True, exist_ok=True)

    # Directories to scan demo code files
    demo_dirs = pathlib.Path("../../demo")
    biharmonic_demo = pathlib.Path("../../demo/biharmonic")

    # Iterate over subdirectories containing demos
    for demo_subdir in demo_dirs.iterdir():
        if demo_subdir.is_dir():
            print(demo_subdir)
            fname = pathlib.Path("/demo_"+demo_subdir.name)
            if demo_subdir == biharmonic_demo:
                print("found")
                print(fname.name)
                # Process each demo using jupytext/myst
                for demo_file in demo_subdir.glob("main.cpp"):
                    # Copy demo files into documentation demo directory
                    shutil.copy(demo_file, demo_doc_dir)
                    cpp_demo = jupytext.read(demo_file)
                    cpp_myst_text = jupytext.writes(cpp_demo, fmt="myst")

                # myst-parser does not process blocks with {code-cell}
                cpp_myst_text = cpp_myst_text.replace("{code-cell}", "cpp")

                # Similarly for python file, dump python myst text in cpp myst
                for pydemo_file in demo_subdir.glob("*.py"):
                    shutil.copy(pydemo_file, demo_doc_dir)
                    python_demo = jupytext.read(pydemo_file)
                    python_myst_text = jupytext.writes(python_demo, fmt="myst")
                    python_myst_text = python_myst_text.replace("{code-cell}", "python")
                    cpp_myst_text = cpp_myst_text.replace("![ufl-code]", python_myst_text)

                myst_file = (demo_doc_dir / fname.name).with_suffix(".md")
                with open(myst_file, "w") as fw:
                    fw.write(cpp_myst_text)

                # There is a posibility to use jupyter-notebooks with C++/C kernels
                # ipynb_file = (demo_doc_dir / fname.name).with_suffix(".ipynb")
                # jupytext.write(cpp_demo, ipynb_file, fmt="ipynb")

if __name__ == "__main__":
    process()
