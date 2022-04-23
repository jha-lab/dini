import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
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
					default='MACR',
                    help="corruption strategy from ['MACR']")
args = parser.parse_args()