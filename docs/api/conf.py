"""Sphinx configuration for Responsive Fine-Tuner."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

project = "Responsive Fine-Tuner"
author = "Responsive Fine-Tuner contributors"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"
release = os.getenv("RFT_VERSION", "0.1.0")
version = release

extensions = [
    "myst_parser",
    """Sphinx configuration for Responsive Fine-Tuner."""

    from __future__ import annotations

    import os
    import sys
    from datetime import datetime
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    project = "Responsive Fine-Tuner"
    author = "Responsive Fine-Tuner contributors"
    current_year = datetime.now().year
    copyright = f"{current_year}, {author}"
    release = os.getenv("RFT_VERSION", "0.1.0")
    version = release

    extensions = [
        "myst_parser",
        "sphinx.ext.autodoc",
        "sphinx.ext.autosummary",
        "sphinx.ext.napoleon",
        "sphinx.ext.viewcode",
    ]

    autosummary_generate = True
    add_module_names = False
    napoleon_google_docstring = True
    napoleon_numpy_docstring = True
    napoleon_use_param = True
    napoleon_use_rtype = True
    autodoc_member_order = "bysource"
    autodoc_typehints = "description"
    autodoc_mock_imports = ["torch", "transformers", "gradio", "peft", "trl", "plotly"]

    source_suffix = {
        """Sphinx configuration for Responsive Fine-Tuner."""

        from __future__ import annotations

        import os
        import sys
        from datetime import datetime
        from pathlib import Path

        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        project = "Responsive Fine-Tuner"
        author = "Responsive Fine-Tuner contributors"
        current_year = datetime.now().year
        copyright = f"{current_year}, {author}"
        release = os.getenv("RFT_VERSION", "0.1.0")
        version = release

        extensions = [
            "myst_parser",
            "sphinx.ext.autodoc",
            "sphinx.ext.autosummary",
            "sphinx.ext.napoleon",
            "sphinx.ext.viewcode",
        ]

        autosummary_generate = True
        add_module_names = False
        napoleon_google_docstring = True
        napoleon_numpy_docstring = True
        napoleon_use_param = True
        napoleon_use_rtype = True
        autodoc_member_order = "bysource"
        autodoc_typehints = "description"
        autodoc_mock_imports = ["torch", "transformers", "gradio", "peft", "trl", "plotly"]

        source_suffix = {
            ".rst": "restructuredtext",
            ".md": "markdown",
        }

        templates_path = ["_templates"]
        html_static_path = ["_static"]
        exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

        html_theme = "sphinx_rtd_theme"
        html_theme_options = {
            "collapse_navigation": False,
            "style_external_links": True,
        }

        html_title = "Responsive Fine-Tuner API"
