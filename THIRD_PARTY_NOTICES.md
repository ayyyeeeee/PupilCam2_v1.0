# Third-party components

This package retains the user's existing application architecture and is not an official Pupil Labs product.

## Runtime

CPython 3.10.11 Windows amd64 components were downloaded from
https://www.python.org/ftp/python/3.10.11/amd64/ and administratively extracted,
then the pinned wheels in requirements.txt were installed. Runtime binaries are
unmodified. Python Software Foundation signatures were verified on the official
installer. Python's LICENSE.txt and Tcl/Tk license files are retained in runtime.zip.

Dependency wheels originate from PyPI. Their original license notices and
*.dist-info metadata remain inside runtime/Lib/site-packages. Version pins are in
requirements.txt. Packages include NumPy, OpenCV, Pillow, pupil-detectors,
pupil-labs-uvc, pye3d, sv-ttk, msgpack, sortedcontainers, pip and setuptools.

## Driver tool

Unmodified Zadig 2.9 is supplied from the official project:
https://github.com/pbatard/libwdi/releases/download/v1.5.1/zadig-2.9.exe

Authenticode signature verified: Akeo Consulting.
License: GPL-3.0-or-later; libwdi: LGPL-3.0-or-later. The corresponding project
source archive (v1.5.1), including license texts and build instructions, is
distributed beside the executable as drivers/libwdi-1.5.1-source.zip, from
https://github.com/pbatard/libwdi/archive/refs/tags/v1.5.1.zip.

Embedded driver components carry their own licenses. See the source archive and
https://zadig.akeo.ie/. No copied local-generated certificate is silently trusted
by this application's setup script. Installing a driver requires the user's
approval in the separate official tool.
