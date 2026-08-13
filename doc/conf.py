# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath('../'))

project = 'ramantune'
copyright = '2026, Simone Innocente'
author = 'Simone Innocente'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary'
]
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
#html_static_path = ['_static']
html_theme = "pydata_sphinx_theme"
add_module_names = False
html_title = 'ramantune'
autodoc_member_order = "bysource"


html_sidebars = {
    #"**": ["sidebar-nav-bs"]
    "**": []
}

html_context = {
    "default_mode": "light"
}

html_theme_options = {
    #"collapse_navigation": True,
    #'"navigation_depth": 1,
    "navbar_end": ["navbar-icon-links","theme-switcher"],
    "icon_links_label": "Quick Links",
    "show_toc_level": 2,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/InnocenteSimone/ramantune/",
            "icon": "fab fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/ramantune/0.0.1/",
            "icon": "fab fa-python",
        }
    ],
}
