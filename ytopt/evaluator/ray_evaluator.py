import logging
import subprocess
import time
from collections import defaultdict, namedtuple
import sys

import ray

from ytopt.evaluator.evaluate import Evaluator

logger = logging.getLogger(__name__)


@ray.remote
def compute_objective(func, x):
    return func(x)

class RayFuture:
    FAIL_RETURN_VALUE = Evaluator.FAIL_RETURN_VALUE

    def __init__(self, func, x):
        self.id_res = compute_objective.remote(func, x)
        self._state = 'active'
        self._result = None

    def _poll(self):
        if not self._state == 'active':
            return

        id_done, _ = ray.wait([self.id_res], num_returns=1, timeout=0.001)

        if len(id_done) == 1:
            try:
                self._result = ray.get(id_done[0])
                self._state = 'done'
            except Exception:
                self._state = 'failed'
        else:
            self._state = 'active'

    def result(self):
        if not self.done:
            self._result = self.FAIL_RETURN_VALUE
        return self._result

    def cancel(self):
        pass # NOT AVAILABLE YET

    @property
    def active(self):
        self._poll()
        return self._state == 'active'

    @property
    def done(self):
        self._poll()
        return self._state == 'done'

    @property
    def failed(self):
        self._poll()
        return self._state == 'failed'

    @property
    def cancelled(self):
        self._poll()
        return self._state == 'cancelled'

class RayEvaluator(Evaluator):
    """Ray

        Args:
            run_function (func): takes one parameter of type dict and returns a scalar value.
            cache_key (func): takes one parameter of type dict and returns a hashable type, used as the key for caching evaluations. Multiple inputs that map to the same hashable key will only be evaluated once. If ``None``, then cache_key defaults to a lossless (identity) encoding of the input dict.
    """
    WaitResult = namedtuple(
        'WaitResult', ['active', 'done', 'failed', 'cancelled'])

    def __init__(self, problem, cache_key=None):
        super().__init__(problem, cache_key)

        proc_info = ray.init()

        self.num_workers = len(ray.nodes())

        logger.info(f"RAY Evaluator will execute: '{self.problem.objective}', proc_info: {proc_info}")

    def _eval_exec(self, x: dict):
        assert isinstance(x, dict)
        future = RayFuture(self.problem.objective, x)
        return future

    @staticmethod
    def _timer(timeout):
        if timeout is None:
            return lambda: True
        else:
            timeout = max(float(timeout), 0.01)
            start = time.time()
            return lambda: (time.time()-start) < timeout

    def wait(self, futures, timeout=None, return_when='ANY_COMPLETED'):
        assert return_when.strip() in ['ANY_COMPLETED', 'ALL_COMPLETED']
        waitall = bool(return_when.strip() == 'ALL_COMPLETED')

        num_futures = len(futures)
        active_futures = [f for f in futures if f.active]
        time_isLeft = self._timer(timeout)

        if waitall:
            def can_exit(): return len(active_futures) == 0
        else:
            def can_exit(): return len(active_futures) < num_futures

        while time_isLeft():
            if can_exit():
                break
            else:
                active_futures = [f for f in futures if f.active]
                time.sleep(0.04)

        if not can_exit():
            raise TimeoutError(f'{timeout} sec timeout expired while '
                               f'waiting on {len(futures)} tasks until {return_when}')

        results = defaultdict(list)
        for f in futures:
            results[f._state].append(f)
        return self.WaitResult(
            active=results['active'],
            done=results['done'],
            failed=results['failed'],
            cancelled=results['cancelled']
        )
