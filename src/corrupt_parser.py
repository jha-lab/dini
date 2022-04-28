import argparse

parser = argparse.ArgumentParser(description='DINI')
parser.add_argument('--dataset', 
					metavar='-d', 
					type=str, 
					required=True,
					default='MSDS',
                    help="dataset from ['MSDS']")
parser.add_argument('--strategy', 
					metavar='-s', 
					type=str, 
					required=False,
					default='MCAR',
                    help="corruption strategy from ['MCAR']")
args = parser.parse_args()