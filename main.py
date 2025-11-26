import argparse

from src import DataProtectionPipeline

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str)
    args = parser.parse_args()

    pipeline = DataProtectionPipeline(config_path=args.config)
    pipeline.run()