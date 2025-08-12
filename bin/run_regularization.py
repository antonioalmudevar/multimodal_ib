import argparse

from src.experiments.disentanglement import RegularizationExperiment


def parse_args():

    parser = argparse.ArgumentParser(description='Train Encoder')

    parser.add_argument(
        'config_file', 
        nargs='?', 
        type=str,
        help='configuration file'
    )

    parser.add_argument(
        '-b', '--beta', 
        type=float, 
        default=0., 
        help='seed for initialization'
    )

    parser.add_argument(
        '-s', '--seed', 
        type=int, 
        default=42, 
        help='seed for initialization'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    experiment = RegularizationExperiment(
        **vars(args), wandb_key='ad6dfde6b67458b23b722ca23221f8d82d3cf713')
    experiment.run_all()