"""
The ``search`` module brings a modular way to implement new search algorithms and two sub modules. One is for hyperparameter search ``ytopt.search.hps`` and one is for neural architecture search ``ytopt.search.nas``.
The ``Search`` class is abstract and has different subclasses such as: ``ytopt.search.ambs`` and ``ytopt.search.ga``.
"""

from ytopt.search.search import Search

__all__ = ['Search']
