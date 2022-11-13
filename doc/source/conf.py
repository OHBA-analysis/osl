# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))


# -- prep auto generated api -------------------------------------------------

# Can assume we're in osl/doc/source

import osl
from inspect import getmembers, isfunction
mne_wrappers = [ff[0] for ff in getmembers(osl.preprocessing.mne_wrappers) if isfunction(ff[1])]
osl_wrappers = [ff[0] for ff in getmembers(osl.preprocessing.osl_wrappers) if isfunction(ff[1])]

from jinja2 import Template
with open('api.rst.jinja2') as file_:
    template = Template(file_.read())

api =  template.render(mne_wrappers=mne_wrappers, osl_wrappers=osl_wrappers)

with open(os.path.join('api.rst'), 'w') as f:
    f.write(api);

# -- Project information -----------------------------------------------------

project = 'osl'
copyright = '2021, OMG'
author = 'OMG'

# The full version, including alpha/beta/rc tags
release = '0.0.1dev'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_gallery.gen_gallery',
    'numpydoc'
]

autodoc_default_flags = ['members']
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

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
html_static_path = ['_static']

# -- Sphinx Options ---------------------------------------------------------

sphinx_gallery_conf = {
     'examples_dirs': 'tutorials',   # path to your example scripts
     'gallery_dirs': 'build_tutorials',  # path to where to save gallery generated output
     'filename_pattern': '/osl_tutorial_',
}


intersphinx_mapping = {'mne': ('https://mne.tools/stable', None)}
