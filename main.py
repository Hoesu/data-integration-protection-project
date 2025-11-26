import argparse

from src import DataProtectionPipeline, load_config, setup_logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging()

    pipeline = DataProtectionPipeline(config)
    pipeline.run()
