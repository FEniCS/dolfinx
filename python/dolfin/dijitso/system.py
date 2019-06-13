# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016 Martin Sandve Alnæs
#
# This file is part of DIJITSO.
#
# DIJITSO is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DIJITSO is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DIJITSO. If not, see <http://www.gnu.org/licenses/>.

"""Utilities for interfacing with the system."""

import os
import errno
import io
import gzip
import shutil
import stat
import uuid
import re
from glob import glob
import tempfile
import subprocess

from dolfin.dijitso.log import warning


def _get_status_output_subprocess(cmd, input=None, cwd=None, env=None):
    """Replacement for commands.getstatusoutput which does not work on Windows (or Python 3)."""
    if isinstance(cmd, str):
        cmd = cmd.strip().split()
    pipe = subprocess.Popen(cmd, shell=False, cwd=cwd, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output, errout) = pipe.communicate(input=input)
    assert not errout
    status = pipe.returncode
    if isinstance(output, bytes):
        output = output.decode('utf-8')
    return (status, output)


def _get_status_output_system(cmd):
    """Replacement for commands.getstatusoutput which does not work on Windows (or Python 3)."""
    if not isinstance(cmd, str):
        cmd = " ".join(cmd)
    # Default return values
    status = 1
    output = "not run"
    # Execute cmd with redirection to temporary file
    with tempfile.NamedTemporaryFile(delete=True) as f:
        cmd += ' > ' + f.name + ' 2>&1'
        # NOTE: Possibly OFED-fork-safe, tests needed!
        status = os.system(cmd)
        output = f.read()
    if isinstance(output, bytes):
        output = output.decode('utf-8')
    return (status, output)


# Choose method for calling external programs. Use subprocess by default.
_call_method = "SUBPROCESS"
_call_method = os.environ.get("DIJITSO_SYSTEM_CALL_METHOD", _call_method)

if _call_method == "OS_SYSTEM":
    get_status_output = _get_status_output_system
else:
    get_status_output = _get_status_output_subprocess


def make_executable(filename):
    "Make script executable by setting user eXecutable bit."
    permissions = stat.S_IMODE(os.lstat(filename).st_mode)
    os.chmod(filename, permissions | stat.S_IXUSR)


def make_dirs(path):
    """Creates a directory (tree).

    Ignores error if the directory already exists.
    """
    try:
        os.makedirs(path)
    except os.error as e:
        if e.errno != errno.EEXIST:
            raise


def rename_file(src, dst):
    """Rename a file.

    Ignores error if the destination file exists.
    """
    try:
        os.rename(src, dst)
    except os.error as e:
        # Windows may trigger on existing destination
        if e.errno not in errno.EEXIST:
            raise


def try_rename_file(src, dst):
    """Try to rename a file.

    NB! Ignores error if the SOURCE doesn't exist or the destination already exists.
    """
    try:
        os.rename(src, dst)
    except os.error as e:
        # Windows may trigger on existing destination,
        # everyone triggers on missing source
        if e.errno not in (errno.ENOENT, errno.EEXIST):
            raise


def try_copy_file(src, dst):
    """Try to copy a file.

    NB! Ignores any error.
    """
    try:
        shutil.copy(src, dst)
    except Exception:
        pass


def try_delete_file(filename):
    """Try to remove a file.

    Ignores error if filename doesn't exist.
    """
    try:
        os.remove(filename)
    except os.error as e:
        if e.errno != errno.ENOENT:
            raise


def gzip_file(filename):
    """Gzip a file.

    New file gets .gz extension added.

    Does nothing if the .gz file already exists.

    Original file is never touched.
    """
    # Expecting source file to be there
    if not os.path.exists(filename):
        return None
    # Avoid doing work if file is already there
    gz_filename = filename + ".gz"
    if not os.path.exists(gz_filename):
        # Write gzipped contents to a temp file
        tmp_filename = filename + "-tmp-" + uuid.uuid4().hex + ".gz"
        with io.open(filename, "rb") as f_in, gzip.open(tmp_filename, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        # Safe move to target filename, other processes may compete here
        lockfree_move_file(tmp_filename, gz_filename)
    return gz_filename


def gunzip_file(gz_filename):
    """Gunzip a file."""
    assert gz_filename[-3:] == ".gz"
    filename = gz_filename[:-3]
    # Write gzipped contents to a temp file
    tmp_filename = filename + "-tmp-" + uuid.uuid4().hex
    with gzip.open(gz_filename, "rb") as f_in, io.open(tmp_filename, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    # Safe move to target filename, other processes may compete here
    lockfree_move_file(tmp_filename, filename)
    try_delete_file(gz_filename)
    return filename


def read_textfile(filename):
    """Try to read file content, if necessary unzipped
    from filename.gz, return None if not found."""
    if not os.path.exists(filename):
        filename = filename + ".gz"
    if not os.path.exists(filename):
        content = None
    else:
        if filename.endswith(".gz"):
            content = b""
            with gzip.open(filename, "rb") as f:
                content = f.read()
            content = content.decode("utf-8")
        else:
            with io.open(filename, "r", encoding="utf-8") as f:
                content = f.read()
    return content


def store_textfile(filename, content):
    """Store content to filename without race conditions.

    Works by first writing to a unique temp file and then
    moving to final destination.

    Handles both bytes and unicode.
    """
    # Generate a unique temporary filename in same directory as the target file
    ui = uuid.uuid4().hex
    tmp_filename = "%s.%s" % (filename, ui)

    # Write the text to a temporary file
    if isinstance(content, bytes):
        # content is already bytes, write raw
        f = io.open(tmp_filename, "wb")
    else:
        f = io.open(tmp_filename, "w", encoding="utf8")

    with f:
        f.write(content)

    # Safely move file to target filename
    lockfree_move_file(tmp_filename, filename)

    return filename


def move_file(srcfilename, dstfilename):
    """Move or copy a file."""
    assert os.path.exists(srcfilename)
    shutil.move(srcfilename, dstfilename)
    assert not os.path.exists(srcfilename)
    assert os.path.exists(dstfilename)


def lockfree_move_file(src, dst):
    """Lockfree and portable nfs safe file move operation.

    If target filename exists with different content,
    will move it to filename.old and emit a warning.

    Taken from textual description at
    http://stackoverflow.com/questions/11614815/a-safe-atomic-file-copy-operation
    """
    return _lockfree_move_file(src, dst, False)


def _lockfree_move_file(src, dst, recurse):
    if not os.path.exists(src):
        if recurse:
            return
        else:
            raise RuntimeError("Source file does not exist.")

    dst_exists = os.path.exists(dst)
    if dst_exists and recurse:
        warning("Backup file exists, overwriting.")
    elif dst_exists:
        # Destination file exists
        with io.open(src, "rb") as f:
            s = f.read()
        with io.open(dst, "rb") as f:
            d = f.read()

        # Uncompress gzipped files, as they contain a timestamp
        # and will otherwise always differ
        if (src[-3:] == '.gz'):
            s = gzip.decompress(s)
            d = gzip.decompress(d)

        if s == d:
            # Files are the same, just delete src instead of moving
            try_delete_file(src)
            return
        # Files differ, backup old file before moving file over it
        backup = dst + ".old"
        warning("Moving new file over differing existing file:\n"
                "src: %s\ndst: %s\nbackup: %s" % (src, dst, backup))
        _lockfree_move_file(dst, backup, True)

    def priv(j):
        return dst + ".priv." + str(j)

    def pub(j):
        return dst + ".pub." + str(j)

    # Create a universally unique 128 bit integer id
    ui = uuid.uuid4().int

    # Move or copy file onto the target filesystem
    move_file(src, priv(ui))

    # Atomic rename to make file visible to competing processes
    rename_file(priv(ui), pub(ui))

    # Find uuids of competing files
    n = len(pub("*")) - 1
    uuids = sorted(int(fn[n:]) for fn in glob(pub("*")))

    # Try to delete all files with larger uuids
    for i in uuids:
        if i > ui:
            try_delete_file(pub(i))
    for i in uuids:
        if i < ui:
            # Our file is the one with a larger uuid
            try_delete_file(pub(ui))
            # Cooperate on handling uuid i
            ui = i

    if os.path.exists(dst):
        # If somebody else beat us to it, delete our file
        try_delete_file(pub(ui))
    else:
        # Otherwise do an atomic rename to make our file final
        try_rename_file(pub(ui), dst)
    if os.path.exists(src):
        if recurse:
            # Somebody already generated new dest file which we just backed up
            pass
        else:
            raise RuntimeError("Source file should not exist at this point!")
    if not os.path.exists(dst):
        raise RuntimeError("Destination file should exist at this point!")


def ldd(libname):
    """Run the ldd system tool on libname.

    Returns output as a dict {basename: fullpath} with all
    dynamic library dependencies and their resolution path.

    This is a debugging tool and may fail if ldd is not
    available or behaves differently on this system.
    """
    status, output = get_status_output(["ldd", libname])
    libraries = {}
    for line in output.splitlines():
        match = re.match(r'(.*)=>([^(]*)(.*)', line)
        if match:
            dlib = match.group(1).strip()
            dlibpath = match.group(2).strip()
            address = match.group(3).strip()
            if address:
                # Path can be empty for system libs
                assert dlibpath == "" or os.path.exists(dlibpath)
                libraries[dlib] = dlibpath
            else:
                assert not os.path.exists(dlibpath)
                libraries[dlib] = None
    return libraries
