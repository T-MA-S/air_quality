"""Main entry point for the application."""

import argparse
from datetime import datetime, timedelta

from src.etl.pipeline import ETLPipeline
from src.data.models import CITIES, get_city_by_name
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Air Quality ETL Pipeline")
    parser.add_argument(
        "--cities",
        nargs="+",
        help="List of cities to process (default: all)",
        default=None,
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to extract (default: 7)",
    )
    parser.add_argument(
        "--date-from",
        type=str,
        help="Start date (YYYY-MM-DD)",
        default=None,
    )
    parser.add_argument(
        "--date-to",
        type=str,
        help="End date (YYYY-MM-DD)",
        default=None,
    )

    args = parser.parse_args()

    # Determine date range
    if args.date_from and args.date_to:
        date_from = datetime.fromisoformat(args.date_from)
        date_to = datetime.fromisoformat(args.date_to)
    else:
        date_to = datetime.now()
        date_from = date_to - timedelta(days=args.days)

    # Select cities
    if args.cities:
        cities = [get_city_by_name(city) for city in args.cities]
        cities = [c for c in cities if c is not None]
        if not cities:
            logger.error("No valid cities found")
            return
    else:
        cities = CITIES

    logger.info(f"Running ETL for {len(cities)} cities from {date_from} to {date_to}")

    # Run pipeline
    pipeline = ETLPipeline(cities=cities)
    results = pipeline.run(date_from=date_from, date_to=date_to)

    logger.info(f"ETL completed: {results}")


if __name__ == "__main__":
    main()

