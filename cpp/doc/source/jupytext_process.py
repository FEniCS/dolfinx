# Copyright (C) 2017-2023 Garth N. Wells, Jack S. Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Convert C++ demos in Jupytext 'light' into MyST markdown."""

import pathlib
import re
import shutil

import jupytext

# MyST roles and inline math, which are not rendered in code cells
_MYST_MARKUP = re.compile(r"\{(cpp|py|doc|download)[:}]|\$[^$\s][^$]*\$")


def check_code_cells(notebook, path: pathlib.Path, comment: str) -> list[str]:
    """Find code cell comments that contain MyST markup.

    A comment is rendered as Markdown only if it is a separate cell.
    Comments indented in a function body, or directly followed by code,
    stay in a code cell and their MyST markup is shown verbatim.

    Args:
        notebook: Jupytext notebook.
        path: Demo file, for error messages.
        comment: Line comment prefix.

    Returns:
        Error messages, one per offending line.
    """
    errors = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for line in cell.source.splitlines():
            text = line.strip()
            if text.startswith(comment) and _MYST_MARKUP.search(text):
                errors.append(f"{path}: {text}")
    return errors


def process():
    """Convert C++ demos in Jupytext 'light' into MyST markdown.

    Uses Jupytext. These files can then be included in Sphinx
    documentation.
    """
    # Make demo doc directory
    demo_doc_dir = pathlib.Path("./demos")
    demo_doc_dir.mkdir(parents=True, exist_ok=True)

    # Directories to scan demo code files
    demo_dirs = pathlib.Path("../../demo")

    errors = []

    # Iterate over subdirectories containing demos
    for demo_subdir in demo_dirs.iterdir():
        if demo_subdir.is_dir():
            fname = pathlib.Path("/demo_" + demo_subdir.name)
            demo_doc_subdir = demo_doc_dir / fname.name
            demo_doc_subdir.mkdir(parents=True, exist_ok=True)
            # Process each demo using jupytext/myst
            for demo_file in demo_subdir.glob("main.cpp"):
                # Copy demo files into documentation demo directory
                shutil.copy(demo_file, demo_doc_subdir)
                cpp_demo = jupytext.read(demo_file)
                errors += check_code_cells(cpp_demo, demo_file, "//")
                cpp_myst_text = jupytext.writes(cpp_demo, fmt="myst")

            # myst-parser does not process blocks with {code-cell}
            cpp_myst_text = cpp_myst_text.replace("{code-cell}", "cpp")

            # Similarly for python file, dump python myst text in cpp myst
            for pydemo_file in demo_subdir.glob("*.py"):
                shutil.copy(pydemo_file, demo_doc_subdir)
                python_demo = jupytext.read(pydemo_file)
                errors += check_code_cells(python_demo, pydemo_file, "#")
                python_myst_text = jupytext.writes(python_demo, fmt="myst")
                python_myst_text = python_myst_text.replace("{code-cell}", "python")
                cpp_myst_text = cpp_myst_text.replace("![ufl-code]", python_myst_text)

            for cmake_file in demo_subdir.glob("CMakeLists.txt"):
                shutil.copy(cmake_file, demo_doc_subdir)

            myst_file = (demo_doc_dir / fname.name).with_suffix(".md")
            with open(myst_file, "w") as fw:
                fw.write(cpp_myst_text)

            # There is a possibility to use jupyter-notebooks with C++/C kernels
            # ipynb_file = (demo_doc_dir / fname.name).with_suffix(".ipynb")
            # jupytext.write(cpp_demo, ipynb_file, fmt="ipynb")

    if errors:
        raise RuntimeError(
            "MyST markup in demo code cell comments, where it is not rendered. "
            "Move the text to a Markdown cell (a comment block at column 0, "
            "separated from code by blank lines) or remove the markup:\n" + "\n".join(errors)
        )


if __name__ == "__main__":
    process()
