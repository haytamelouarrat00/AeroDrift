import polars as pl

def load_and_clean_data(file_path: str) -> pl.DataFrame:
    cols = ['engine_id', 'cycle', 'op1', 'op2', 'op3'] + [f'sensor_{i}' for i in range(1, 22)]
    # Polars read_csv does not support regex separator like '\s+'. 
    # We use ' ' and it will read extra null columns which we discard.
    df = pl.read_csv(file_path, separator=" ", has_header=False)
    
    # Keep only the first 26 columns and rename them
    df = df.select(df.columns[:26])
    df = df.rename(dict(zip(df.columns, cols)))
    return df.drop_nulls()

def add_rul(df: pl.DataFrame) -> pl.DataFrame:
    max_cycles = (
        df
        .group_by("engine_id")
        .agg(pl.col("cycle").max().alias("max_cycle"))
    )

    return (
        df
        .join(max_cycles, on="engine_id")
        .with_columns((pl.col("max_cycle") - pl.col("cycle")).alias("RUL"))
    )

def add_rolling_features(df: pl.DataFrame, window_size: int = 15) -> pl.DataFrame:
    # We use .over("engine_id") to compute rolling features per engine independently.
    # The data is assumed to be ordered by cycle for each engine.
    return df.with_columns(
        [
            pl.col(c).rolling_mean(window_size=window_size).over("engine_id").alias(f"{c}_rolling_mean")
            for c in ["sensor_4", "sensor_11"]
        ] + [
            pl.col(c).rolling_std(window_size=window_size).over("engine_id").alias(f"{c}_rolling_std")
            for c in ["sensor_4", "sensor_11"]
        ]
    )
