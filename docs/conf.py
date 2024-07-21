# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import pathlib
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(1, os.path.dirname(
    os.path.abspath("../")) + os.sep + "stream")

project = "stream"
author = "Anton Frederik Thielmann"

VERSION_PATH = "../stream/__version__.py"
with open(VERSION_PATH) as f:
    VERSION = f.readlines()[-1].split()[-1].strip("\"'")
release = VERSION

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "nbsphinx",
    'myst_nb',
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    # "myst_parser",
    "mdinclude",
    "sphinx_autodoc_typehints",
]

autodoc_mock_imports = [
    "numpy",
    "lightning",
    "torch",
    "torchmetrics",
    "pytorch_lightning",
    "pandas",
    "sklearn",
    "tqdm",
    "transformers",
    "sentence_transformers",
    "umap",
    "hdbscan",
    "wordcloud",
    "plotly",
    "matplotlib",
    "gensim",
    "nltk",
    "langdetect",
    "octis",
    "loguru",
    "scipy",
    "community",
    "networkx",
    "pyarrow"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = [".rst", ".ipynb"]
source_suffix = [".rst", ".ipynb", ".md"]

# The master toctree document.
master_doc = "index"

# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "_templates"]

# The reST default role (single back ticks `dict`) cross links to any code
# object (including Python, but others as well).
default_role = "literal"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ['_static']
html_css_files = ['css/stream_theme.css']
# html_css_files = ['custom.css']
# html_js_files = ['custom.js']

html_theme_options = {
    "globaltoc_collapse": False,
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "images/logos/STREAM_2.jpg"

# Override the Sphinx default title that appends `documentation`
html_title = f"{project}"
# Format of the last updated section in the footer
html_last_updated_fmt = "%Y-%m-%d"

# -- Options for autodoc ------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "exclude-members": "set_output",
}

# generate autosummary even if no references
autosummary_generate = True

# -- Options for numpydoc -----------------------------------------------------

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69

numpydoc_show_class_members = False
