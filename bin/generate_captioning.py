import argparse

from src.experiments.captioning import GenerateExperiment

def parse_args():

    parser = argparse.ArgumentParser(description='Train Encoder')

    parser.add_argument(
        'config_file', 
        nargs='?', 
        type=str,
        help='configuration file'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    experiment = GenerateExperiment(
        **vars(args), wandb_key='ad6dfde6b67458b23b722ca23221f8d82d3cf713')
    experiment.run_all()