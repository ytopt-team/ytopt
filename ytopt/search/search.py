import argparse
from pprint import pformat
import logging
from ytopt.search import util
from ytopt.evaluator.evaluate import Evaluator

logger = logging.getLogger(__name__)


class Namespace:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v


class Search:
    """Abstract representation of a black box optimization search.

    A search comprises 3 main objects: a problem, a run function and an evaluator:
        The `problem` class defines the optimization problem, providing details like the search domain.  (You can find many kind of problems in `ytopt.benchmark`)
        The `run` function executes the black box function/model and returns the objective value which is to be optimized.
        The `evaluator` abstracts the run time environment (local, supercomputer...etc) in which run functions are executed.

    Args:
        problem (str): Module path to the Problem instance you want to use for the search (e.g. ytopt.benchmark.hps.polynome2.Problem).
        run (str): Module path to the run function you want to use for the search (e.g. ytopt.benchmark.hps.polynome2.run).
        evaluator (str): value in ['balsam', 'subprocess', 'processPool', 'threadPool'].
    """

    def __init__(self, problem, evaluator, cache_key=None, max_evals=100, eval_timeout_minutes=None, redis_address=None, **kwargs):
        settings = kwargs
        settings['problem'] = problem
        settings['evaluator'] = evaluator
        settings['cache_key'] = cache_key



        self.problem = util.generic_loader(problem, 'Problem')

        if cache_key is None:
            self.evaluator = Evaluator.create(self.problem, method=evaluator, redis_address=redis_address)
        else:
            self.evaluator = Evaluator.create(
                self.problem, method=evaluator, cache_key=cache_key, redis_address=redis_address)

        self.max_evals = max_evals
        self.eval_timeout_minutes = eval_timeout_minutes

        self.num_workers = self.evaluator.num_workers

        logger.info(f'Options: {pformat(dict(settings), indent=4)}')
        logger.info(f'Hyperparameter space definition: {pformat(self.problem.input_space, indent=4)}')
        logger.info(f'Created "{evaluator}" evaluator')
        logger.info(f'Evaluator: num_workers is {self.num_workers}')

    def main(self):
        raise NotImplementedError

    @classmethod
    def parse_args(cls, arg_str=None):
        base_parser = cls._base_parser()
        parser = cls._extend_parser(base_parser)
        if arg_str is not None:
            return parser.parse_args(arg_str)
        else:
            return parser.parse_args()

    @staticmethod
    def _extend_parser(base_parser):
        raise NotImplementedError

    @staticmethod
    def _base_parser():
        parser = argparse.ArgumentParser(conflict_handler='resolve')
        parser.add_argument("--problem",
                            default="ytopt.benchmark.ackley.problem.Problem",
                            help="Module path to the Problem instance you want to use for the search (e.g. ytopt.benchmark.ackley.problem.Problem)."
                            )
        parser.add_argument('--max-evals',
                            type=int, default=100,
                            help='maximum number of evaluations'
                            )
        parser.add_argument('--eval-timeout-minutes',
                            type=int,
                            default=4096,
                            help="Kill evals that take longer than this"
                            )
        parser.add_argument('--evaluator',
                            default='subprocess',
                            choices=['balsam', 'subprocess', 'ray'],
                            help="The evaluator is an object used to run the model."
                            )
        parser.add_argument('--redis-address', type=str, default=None,
                            help="The redis-address for Ray-Worker when ray evaluator is choosen")
        return parser
