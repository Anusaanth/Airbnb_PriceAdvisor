# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Load raw data
    output_filepath = '../data/raw/euro_data.csv'
    athens_weekdays = pd.read_csv('../data/raw/athens_weekdays.csv')
    athens_weekends = pd.read_csv('../data/raw/athens_weekends.csv')
    athens = comb(athens_weekdays, "weekdays", athens_weekends, "weekends", "athens")
    
    barcelona_weekdays = pd.read_csv('../data/raw/barcelona_weekdays.csv')
    barcelona_weekends = pd.read_csv('../data/raw/barcelona_weekends.csv')
    barcelona = comb(barcelona_weekdays, "weekdays", barcelona_weekends, "weekends", "barcelona")

    berlin_weekdays = pd.read_csv('../data/raw/berlin_weekdays.csv')
    berlin_weekends = pd.read_csv('../data/raw/berlin_weekends.csv')
    berlin = comb(berlin_weekdays, "weekdays", berlin_weekends, "weekends", "berlin")

    lisbon_weekdays = pd.read_csv('../data/raw/lisbon_weekdays.csv')
    lisbon_weekends = pd.read_csv('../data/raw/lisbon_weekends.csv')
    lisbon = comb(lisbon_weekdays, "weekdays", lisbon_weekends, "weekends", "lisbon")

    london_weekdays = pd.read_csv('../data/raw/london_weekdays.csv')
    london_weekends = pd.read_csv('../data/raw/london_weekends.csv')
    london = comb(london_weekdays, "weekdays", london_weekends, "weekends", "london")

    paris_weekdays = pd.read_csv('../data/raw/paris_weekdays.csv')
    paris_weekends = pd.read_csv('../data/raw/paris_weekends.csv')
    paris = comb(paris_weekdays, "weekdays", paris_weekends, "weekends", "paris")

    rome_weekdays = pd.read_csv('../data/raw/rome_weekdays.csv')
    rome_weekends = pd.read_csv('../data/raw/rome_weekends.csv')
    rome = comb(rome_weekdays, "weekdays", rome_weekends, "weekends", "rome")

    vienna_weekdays = pd.read_csv('../data/raw/vienna_weekdays.csv')
    vienna_weekends = pd.read_csv('../data/raw/vienna_weekends.csv')
    vienna = comb(vienna_weekdays, "weekdays", vienna_weekends, "weekends", "vienna")

    cities = [athens,barcelona,berlin,lisbon,london,paris,rome,vienna]

    euro_data = pd.concat(cities, ignore_index=True)

    euro_data.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

#Function to combine weekend and weekdays dataset for each city
def comb(csv, col, csv2, col2, city):
    csv["week_time"] = col
    csv2["week_time"] = col2

    merg = pd.concat([csv,csv2])
    merg["city"] = city

    return merg

