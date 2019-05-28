import json
import logging
import os
from io import StringIO

from balsam.core.models import ApplicationDefinition as AppDef
from balsam.launcher import dag
from balsam.launcher.async import FutureTask
from balsam.launcher.async import wait as balsam_wait

from ytopt.evaluator.evaluate import Evaluator
from django.core.exceptions import ObjectDoesNotExist
from django.db import transaction

logger = logging.getLogger(__name__)

LAUNCHER_NODES = int(os.environ.get('BALSAM_LAUNCHER_NODES', 1))


class BalsamEvaluator(Evaluator):
    """Evaluator using balsam software.

    Documentation to balsam : https://balsam.readthedocs.io
    This class helps us to run task on HPC systems with more flexibility and ease of use.

    Args:
        run_function (func): takes one parameter of type dict and returns a scalar value.
        cache_key (func): takes one parameter of type dict and returns a hashable type, used as the key for caching evaluations. Multiple inputs that map to the same hashable key will only be evaluated once. If ``None``, then cache_key defaults to a lossless (identity) encoding of the input dict.
    """

    def __init__(self, problem, cache_key=None):
        super().__init__(problem, cache_key)
        self.id_key_map = {}
        self.num_workers = max(1, LAUNCHER_NODES*self.WORKERS_PER_NODE - 2)
        logger.info("Balsam Evaluator instantiated")
        logger.debug(f"LAUNCHER_NODES = {LAUNCHER_NODES}")
        logger.debug(f"WORKERS_PER_NODE = {self.WORKERS_PER_NODE}")
        logger.debug(f"Total number of workers: {self.num_workers}")
        self._init_app()
        logger.info(f"Backend runs will execute function: {self.appName}")
        self.transaction_context = transaction.atomic

    def wait(self, futures, timeout=None, return_when='ANY_COMPLETED'):
        return balsam_wait(futures, timeout=timeout, return_when=return_when)

    def _init_app(self):
        self.appName = self.problem.app_name if self.problem.app_name is not None else self._executable
        try:
            app = AppDef.objects.get(name=self.appName)
        except ObjectDoesNotExist:
            logger.info(
                f"ApplicationDefinition did not exist for {self.appName}; creating new app in BalsamDB")
            app = AppDef(name=self.appName, executable=self._executable)
            app.save()
        finally:
            logger.info(
                f"BalsamEvaluator will use existing app {self.appName}: {self._executable}")

    def _eval_exec(self, x):
        jobname = f"task{self.counter}"
        # args = f"'{self.encode(x)}'"
        args = self.problem.args_format(x.values())
        pb_res = self.problem.resources
        envs = f""
        resources = {
            # 'num_nodes': 1 if x.get('num_nodes') is None else x['num_nodes'],
            'num_nodes': x.get('num_nodes') or pb_res.get('num_nodes'),
            # 'ranks_per_node': 1 if x.get('ranks_per_node') is None else x['ranks_per_node'],
            'ranks_per_node': x.get('ranks_per_node') or pb_res.get('ranks_per_node'),
            # 'threads_per_rank': 64 if x.get('threads_per_rank') is None else x['threads_per_rank'],
            'threads_per_rank': x.get('threads_per_rank') or pb_res.get('threads_per_rank'),
            # 'threads_per_core': 1, if x.get('threads_per_core') is None else x['threads_per_core'],
            'threads_per_core': x.get('threads_per_core') or pb_res.get('threads_per_core'),
            # 'cpu_binding': 'none' if x.get('cpu_binding') is None else x['cpu_binding'],
            'cpu_binding': x.get('cpu_binding') or pb_res.get('cpu_binding')
            'node_packing_count': self.WORKERS_PER_NODE,
        }
        for key in resources:
            if key in x:
                resources[key] = x[key]

        if dag.current_job is not None:
            wf = dag.current_job.workflow
        else:
            wf = self.appName
        task = dag.add_job(
            name=jobname,
            workflow=wf,
            application=self.appName,
            args=args,
            environ_vars=envs,
            **resources
        )
        logger.debug(f"Created job {jobname}")
        logger.debug(f"Args: {args}")

        future = FutureTask(task, self._on_done, fail_callback=self._on_fail)
        future.task_args = args
        return future

    @staticmethod
    def _on_done(job):  # def _on_done(job, process_data):
        output = job.read_file_in_workdir(f'{job.name}.out')
        # process_data(job)
        #args = job.args
        #args = args.replace("\'", "")
        # with open('test.json', 'w') as f:
        #    f.write(args)
        #
        # with open('test.json', 'r') as f:
        #    args = json.load(f)
        output = Evaluator._parse(output)
        #job.data['reward'] = output
        #job.data['arch_seq'] = args['arch_seq']
        #job.data['id_worker'] = args['w']
        # job.save()
        return output

    @staticmethod
    def _on_fail(job):
        logger.info(
            f'Task {job.cute_id} failed; setting objective as float_max')
        return Evaluator.FAIL_RETURN_VALUE
