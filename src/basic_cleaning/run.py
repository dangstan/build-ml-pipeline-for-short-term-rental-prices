#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)
    logger.info('data loaded')

    # Drop outliers
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    logger.info('outliers dropped')

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    logger.info('last_review converted to datetime')
    df.to_csv("clean_sample.csv", index=False)
    logger.info('data saved locally')

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)
    logger.info('cleaned data uploaded To W&B')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help='the name for the input artifact',
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help='the name for the output artifact',
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help='the type for the output artifact',
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help='a description for the output artifact',
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help='the minimum price to consider',
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help='the maximum price to consider',
        required=True
    )


    args = parser.parse_args()

    go(args)
