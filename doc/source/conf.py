# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
import inspect

__location__ = os.path.join(
    os.getcwd(), os.path.dirname(inspect.getfile(inspect.currentframe()))
)

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.join(__location__, "../.."))

# -- Project information -----------------------------------------------------

project = 'OSL'
copyright = '2023, OHBA Analysis Group'
author = 'OHBA Analysis Group'

# The full version, including alpha/beta/rc tags
release = '0.8.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'numpydoc',
]

autodoc_default_flags = ['members']
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates',]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

html_static_path = []

# The master toctree document.
master_doc = 'index'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static',]

# -- Sphinx Options ---------------------------------------------------------

sphinx_gallery_conf = {
     'examples_dirs': 'tutorials',   # path to your example scripts
     'gallery_dirs': 'build_tutorials',  # path to where to save gallery generated output
     'filename_pattern': '/osl_tutorial_',
}

intersphinx_mapping = {'mne': ('https://mne.tools/stable/', None), 
                       'osl': ('https://osl.readthedocs.io/en/improve_docs/', None), 
                       'dask': ('https://distributed.dask.org/en/stable/', None),
                       'sails': ('https://sails.readthedocs.io/en/stable/', None),
                       'matplotlib': ('https://matplotlib.org/stable/', None),}
