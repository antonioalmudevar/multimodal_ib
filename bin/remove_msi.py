import argparse

from src.experiments.disentanglement import RemoveMSIExperiment


def parse_args():

    parser = argparse.ArgumentParser(description='Train Encoder')

    parser.add_argument(
        'config_file', 
        nargs='?', 
        type=str,
        help='configuration file'
    )

    parser.add_argument(
        'missing_factor', 
        nargs='?', 
        type=str,
        help='missing factor of variation'
    )

    parser.add_argument(
        '-s', '--seed', 
        type=int, 
        default=42, 
        help='seed for initialization'
    )

    parser.add_argument(
        '-t', '--temperature', 
        type=float, 
        default=None, 
        help='value of temperature scaling'
    )

    parser.add_argument(
        '-e', '--img_encoder', 
        type=str, 
        default=None,
        help='image encoder'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    experiment = RemoveMSIExperiment(
        **vars(args), wandb_key='ad6dfde6b67458b23b722ca23221f8d82d3cf713')
    experiment.run_all()