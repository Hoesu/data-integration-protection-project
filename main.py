import argparse

import yaml

from src import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str)
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    result_dir = setup_result_directory(config)
    save_config(config, result_dir)
    
    logger = setup_logging(log_path=result_dir)

    pipeline = DataProtectionPipeline(config)
    pipeline.run()
