import argparse

def parse_args():
    parser = argparse.ArgumentParser(
           formatter_class=argparse.ArgumentDefaultsHelpFormatter) 

    parser.add_argument('--hammerblade', default=False, action='store_true',
                        help="Run this on Hammerblade")
    return parser.parse_args()
