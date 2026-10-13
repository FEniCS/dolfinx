# Copyright (C) 2017-2023 Garth N. Wells, Jack S. Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Convert Python demos in Jupytext 'light' into MyST markdown/ipynb."""

import argparse
import os
import pathlib
import runpy
import shutil

import jupytext


def process(run_demos: bool = True):
    """Convert Python demos in Jupytext 'light' into MyST markdown/ipynb.

    Uses Jupytext. These files can then be included in Sphinx
    documentation.

    Demos that save figures are executed so that the images they
    reference exist. This dominates the build time, so it can be
    skipped when the images are already present from an earlier build
    and do not need refreshing.

    Args:
        run_demos: If ``True``, execute demos that save figures to
            (re)generate their images. If ``False``, skip execution and
            leave any existing images in place. A build that skips
            execution without the images already present will report
            missing images, which ``sphinx -W`` turns into errors.
    """
    # Directories to scan
    subdirs = [pathlib.Path("../../demo")]

    # Iterate over subdirectories containing demos
    for subdir in subdirs:
        # Make demo doc directory
        demo_dir = pathlib.Path("./demos")
        demo_dir.mkdir(parents=True, exist_ok=True)

        # Process each demo using jupytext/myst
        for demo in subdir.glob("**/demo*.py"):
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
            shutil.copy(demo, demo_dir)

            # If demo saves matplotlib images, run the demo to create
            # images
            code = demo.read_text()
            if "savefig" in code:
                if run_demos:
                    demo = demo.resolve()
                    here = os.getcwd()
                    os.chdir(demo_dir)
                    runpy.run_path(str(demo))
                    os.chdir(here)
                else:
                    print(f"Skipping execution of {demo.name} (images not refreshed)")

        # Copy images used in demos
        for file in subdir.glob("**/*.png"):
            shutil.copy(file, demo_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-demo-images",
        action="store_true",
        help="do not execute demos that save figures, leaving existing images in place",
    )
    args = parser.parse_args()
    process(run_demos=not args.skip_demo_images)
