import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset', 
					metavar='-d', 
					type=str, 
					required=False,
					default='MSDS',
                    help="dataset from ['MSDS']")
parser.add_argument('--model', 
					metavar='-m', 
					type=str, 
					required=False,
					default='FCN2',
                    help="model name from ['FCN', 'FCN2']")
parser.add_argument('--test', 
					action='store_true', 
					help="test the model")
parser.add_argument('--retrain', 
					action='store_true', 
					help="retrain the model")
parser.add_argument('--model_unc', 
					action='store_true', 
					help="retrain the model")
parser.add_argument('--impute_fraction', 
					metavar='-f', 
					type=float, 
					required=False,
					default=1,
                    help="fraction of data to impute based on uncertainty")
args = parser.parse_args()