#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
import os
import json
import feature_engineer as fe


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="advanced_feature_engineer")
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

    df = fe.interpolating(df)
    df = fe.get_distance(df)
    df,groupers = fe.grouping(df)
    df,dummies = fe.final_adjustments(df)

    boruta_featured = list(fe.reducing_features(df))

    df = df[['last_review','name','price']+boruta_featured]

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    logger.info('last_review converted to datetime')
    df.to_csv("featurized.csv", index=False)
    logger.info('data saved locally')

    if not os.path.exists('groupers'):
        os.makedirs('groupers')
    for i in range(len(groupers)):
        groupers[i].to_csv(f'groupers/grouper_{str(i)}.csv')

    
    with open('dummies.json', 'w') as f:
        json.dump(dummies, f)
    
    with open('boruta_featured.json', 'w') as f:
        json.dump(dummies, f)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_dir('groupers')
    artifact.add_file("dummies.json")
    artifact.add_file("boruta_featured.json")
    artifact.add_file("featurized.csv")
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
