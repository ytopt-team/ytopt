Basic Usage
***********

``ytopt`` is typically run from the command-line in the following example manner::

    python -m ytopt.search.ambs --evaluator ray --problem problem.Problem --max-evals=10 --learner RF

Where:
  * The *search* variant is one of ``ambs`` (*Asynchronous Model-Based Search*) or ``async_search`` (run as an MPI process).
  * The *evaluator* is the method of concurrent evaluations, and can be ``ray`` or ``subprocess``.
  * The *problem* is typically an ``autotune.TuningProblem`` instance. Specify the module path and instance name.
  * ``--max-evals`` is self explanatory.

Depending on the *search* variant chosen, other command-line options may be provided. For example, the ``ytopt.search.ambs`` search
method above was further customized by specifying the ``RF`` learning strategy.

See the `autotune docs`_ for basic information on getting started with creating a ``TuningProblem`` instance.

See the `ConfigSpace docs`_ for guidance on defining input/output parameter spaces for problems.

Otherwise, browse the ``ytopt/benchmark`` directory for an extensive collection of examples.

.. _`autotune docs`: https://github.com/ytopt-team/autotune
.. _`ConfigSpace docs`: https://automl.github.io/ConfigSpace/main/