# Sphinx configuration for rctd-py, following the scverse cookiecutter layout.
# https://www.sphinx-doc.org/page/usage/configuration.html
from datetime import datetime
from importlib.metadata import metadata

info = metadata("rctd-py")
project = info["Name"]
author = info["Author"] or "Paul Gueguen"
copyright = f"{datetime.now():%Y}, {author}"
version = release = info["Version"]
repository_url = "https://github.com/p-gueguen/rctd-py"

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]
needs_sphinx = "7.0"
nitpicky = True  # warn on every unresolved reference; CI builds with -W

html_context = {
    "display_github": True,
    "github_user": "p-gueguen",
    "github_repo": "rctd-py",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "sphinx_click",
    "sphinxext.opengraph",
]

autosummary_generate = True
autodoc_member_order = "bysource"
default_role = "literal"
napoleon_google_docstring = True  # rctd-py docstrings use Args: / Returns:
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
typehints_defaults = "braces"
always_use_bars_union = True

myst_heading_anchors = 6
myst_enable_extensions = ["amsmath", "colon_fence", "deflist", "dollarmath", "html_image", "html_admonition"]
myst_url_schemes = ("http", "https", "mailto")
# Tutorials are executed offline and committed with outputs: Read the Docs has no GPU and would
# re-download the 404 MB Q-matrices on every build.
nb_execution_mode = "off"
nb_output_stderr = "remove"
nb_merge_streams = True

source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.scverse.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}

# index.html / tutorial.html redirect the old GitHub Pages URLs to Read the Docs; plans/ are
# internal design notes.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints", "plans", "*.html"]

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_title = project
html_logo = "logo_max.png"
html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
    "use_issues_button": True,
    "path_to_docs": "docs/",
    "navigation_with_keys": False,
}
pygments_style = "default"

nitpick_ignore = [
    ("py:class", "torch.device"),
]
