# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "InfoVar"
copyright = "2024, Lucas Einig"
author = "Lucas Einig"
release = "0.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

master_doc = "index"


extensions = [
    "sphinx.ext.autodoc",  # Include documentation from docstrings
    "sphinx.ext.coverage",  # Collect doc coverage stats
    "sphinx.ext.doctest",  # Test snippets in the documentation
    "sphinx.ext.extlinks",  # Markup to shorten external links
    "sphinx.ext.inheritance_diagram",  # Include inheritance diagrams
    "sphinx.ext.mathjax",  # Render math via JavaScript
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.todo",  # Support for todo items # .. todo:: directive
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "matplotlib.sphinxext.plot_directive",  # .. plot:: directive for plt.plot
    "myst_parser",
    "sphinx_design",
    "nbsphinx",
    "nbsphinx_link",
]

napoleon_google_docstring = False

autodoc_default_options = {"ignore-module-all": True}
autoclass_content = "both"

# myst_enable_extensions = ["colon_fence"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
