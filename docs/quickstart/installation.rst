Additional Installation Instructions
************************************

Local
=====

``ytopt`` can be installed on your local machine using pip. Doing this in a virtual environment is highly
recommended:

.. code-block:: bash

    git clone https://github.com/ytopt-team/ytopt.git
    cd ytopt/
    pip install -e .

Contribute to documentation
===========================

Installation
------------

.. code-block:: bash

    source activate ENV_NAME
    pip install -U Sphinx
    pip install sphinx_bootstrap_theme

Build
-----

To build the documentation, change to the ``ytopt/docs`` directory and run ``make html``, assuming you have ``make`` installed on your computer.
The results can be viewed in a browser with ``open _build/html/index.html``.

Documentation Architecture
--------------------------

The documentation is made with Sphinx and the following extensions:

============= =============
 Extensions
---------------------------
 Name          Description
============= =============
 autodoc       automatically insert docstrings from modules
 napoleon      inline code documentation
 doctest       automatically test code snippets in doctest blocks
 intersphinx   link between Sphinx documentation of different projects
 todo          write "todo" entries that can be shown or hidden on build
 coverage      checks for documentation coverage
 mathjax       include math, rendered in the browser by MathJax
 ifconfig      conditional inclusion of content based on config values
 viewcode      include links to the source code of documented Python objects
 githubpages   create .nojekyll file to publish the document on GitHub pages
============= =============

Sphinx uses reStructuredText files, click on this `link <https://pythonhosted.org/an_example_pypi_project/sphinx.html>`_ if you want to have an overview 
of the corresponding syntax and mechanism.

Our documentation try to take part of the inline documentation in the code to auto-generate documentation from it. For that reason we highly recommend 
you to follow specific rules when writing inline documentation : https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html.
