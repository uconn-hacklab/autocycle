# Documentation

This documentation is built using *Sphinx* using the *Furo* theme. The markup languages supported are reStructuredText(rst) and MyST Markdown (through myst-parser extension). We currently recommend using MyST as it is a lot similar to traditional markdown.

We are actively working on comprehensive documentation for this project. To build and view the documentation:

1. Install documentation dependencies:
   ```
   pip install -e ".[dev]"
   # OR using uv
   uv pip install ".[dev]"
   ```

2. Build the documentation:
   ```
   cd docs
   sphinx-build -b html . _build/html
   ```

   **OR**

3. For live preview during documentation writing:
   ```
   sphinx-autobuild docs docs/build/_html
   ```
   Then visit localhost in your browser.

   1. To add new documentation, you can either add markdown files to existing directories or create new ones. Each directory should have an `index.md` file that serves as its main page. The root `docs/index.md` contains the master table of contents (toctree) that organizes all documentation. The toctree directive tells Sphinx how to organize the hierarchical structure of the docs. Use our current docs for reference.
   
   2. Use this link for [markup reference](https://pradyunsg.me/furo/reference/).

   3. On merging to main the docs will be deployed to this [page]() within a few seconds.