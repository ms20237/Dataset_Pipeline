import fiftyone as fo
import argparse 
from utils import parse_print_args

# from clearml import maintask
# from clearml import Task


def init():
    """this def is for get parameters that run wants
    """
    
    parser = argparse.ArgumentParser(prog='PROG', usage='%(prog)s [options]')
    parser.add_argument('--arg1',
                        type=str,
                        required=False,
                        help=""
                        )
    
    args = parser.parse_args()
    return args

@parse_print_args
def run(arg1):
    """this def is for run task
    """
    
    
if __name__ == "__main__":

    args = init()
    run(args.arg1)    
    # maintask(run())
    
    
    