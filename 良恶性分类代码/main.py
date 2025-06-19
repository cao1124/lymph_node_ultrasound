import os
import warnings
from args import train_argparser, test_argparser
from trainers.trainer import Trainer
warnings.filterwarnings('ignore')
def _train():
    arg_parser = train_argparser()
    run_args=arg_parser.parse_args()
    trainer = Trainer(run_args)
    trainer.train()

def _test():
    arg_parser = test_argparser()
    run_args = arg_parser.parse_args()
    trainer = Trainer(run_args)
    trainer.test()



if __name__ == '__main__':
    _test()



