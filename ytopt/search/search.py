import argparse
from ytopt.search.utils import saveMetaData, load_from_file, setup_expfolder
from ytopt.evaluate import evaluate

class Search:
    def __init__(self, **kwargs):
        param_dict = kwargs
        self.prob_path = param_dict['prob_path'] #'/Users/pbalapra/Projects/repos/2017/dl-hps/benchmarks/test'
        self.prob_attr = param_dict['prob_attr']
        self.exp_dir = param_dict['exp_dir'] #'/Users/pbalapra/Projects/repos/2017/dl-hps/experiments'
        self.eid = param_dict['exp_id'] #'exp-01'
        self.max_evals = param_dict['max_evals']
        self.max_time = param_dict['max_time']

        #exp_dir = exp_dir+'/'+eid
        self.jobs_dir = self.exp_dir+'/jobs'
        self.results_dir = self.exp_dir+'/results'
        self.results_json_fname = self.exp_dir+'/'+self.eid+'_results.json'
        self.results_csv_fname = self.exp_dir+'/'+self.eid+'_results.csv'
        self.meta_data_json_fname = self.exp_dir+'/'+self.eid+'_metadata.json'
        setup_expfolder(exp_dir=self.exp_dir, job_dir=self.jobs_dir, res_dir=self.results_dir)
        saveMetaData(param_dict, self.meta_data_json_fname)

        self.problem = load_from_file(self.prob_path, self.prob_attr)
        self.problem.checkcfg()
        self.spaceDict = self.problem.space
        self.params = self.problem.params
        self.starting_point = self.problem.starting_point

        self.evaluate = evaluate


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
        parser = argparse.ArgumentParser()
        parser.add_argument('-v', '--version', action='version',
                            version='%(prog)s 0.1')
        parser.add_argument("--prob_path", nargs='?', type=str,
                            default='problems/prob1',
                            help="problem directory")
        parser.add_argument("--prob_attr", type=str,
                            default='problem',
                            help="attribute to load from the file located at prob_path")
        parser.add_argument("--exp_dir", nargs='?', type=str,
                            default='experiments',
                            help="experiments directory")
        parser.add_argument("--exp_id", nargs='?', type=str,
                            default='exp-01',
                            help="experiments id")
        parser.add_argument('--max_evals', action='store', dest='max_evals',
                            nargs='?', const=2, type=int, default='1000',
                            help='maximum number of evaluations')
        parser.add_argument('--max_time', action='store', dest='max_time',
                            nargs='?', const=1, type=float, default='2000',
                            help='maximum time in secs')
        return parser