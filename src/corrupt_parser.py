import argparse

parser = argparse.ArgumentParser(description='DINI')
parser.add_argument('--dataset', 
					metavar='-d', 
					type=str, 
					required=False,
					default='MSDS',
                    help="dataset from ['concrete', 'diamonds', 'energy', 'flights']")
parser.add_argument('--strategy', 
					metavar='-s', 
					type=str, 
					required=False,
					default='MCAR',
                    help="corruption strategy from ['MCAR', 'MAR', 'MNAR', 'MPAR', 'MSAR']")
parser.add_argument('--fraction',
					metavar='-f',
					type=float,
					required=False,
					default=0.1,
					help="fraction of data to corrupt; should be less than 1")
args = parser.parse_args()