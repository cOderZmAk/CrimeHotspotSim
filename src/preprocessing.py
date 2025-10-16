# src/preprocessing.py
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box

LAT_CANDS = ["Latitude", "LAT", "lat", "Y", "Lat"]
LON_CANDS = ["Longitude", "LON", "lon", "X", "Lng", "Long"]
DATE_CANDS = ["Date", "DATE OCC", "DATE_OCC", "Occurred_Date", "Reported_Date", "date", "DATE"]

def _find_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def load_and_clean(raw_csv, processed_dir="../data/processed"):
    os.makedirs(processed_dir, exist_ok=True)
    df = pd.read_csv(raw_csv)

    lat = _find_col(df, LAT_CANDS)
    lon = _find_col(df, LON_CANDS)
    dt  = _find_col(df, DATE_CANDS)
    if not all([lat, lon, dt]):
        raise ValueError(f"Could not detect columns. LAT={lat}, LON={lon}, DATE={dt}")

    df = df.rename(columns={lat: "Latitude", lon: "Longitude", dt: "Date"})
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()
    df["Latitude"]  = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df = df.dropna(subset=["Date"])
    df = df[df["Date"].dt.year >= 2020].copy()

    out = f"{processed_dir}/crime_cleaned.csv"
    df.to_csv(out, index=False)
    return df

def build_grid(gdf_wgs84, grid_size_m=500, processed_dir="../data/processed"):
    utm = gdf_wgs84.estimate_utm_crs()
    gdf_utm = gdf_wgs84.to_crs(utm)

    minx, miny, maxx, maxy = gdf_utm.total_bounds
    pad = grid_size_m
    minx, miny, maxx, maxy = minx - pad, miny - pad, maxx + pad, maxy + pad

    xs = np.arange(minx, maxx, grid_size_m)
    ys = np.arange(miny, maxy, grid_size_m)
    cells = [box(x, y, x + grid_size_m, y + grid_size_m) for x in xs for y in ys]

    grid = gpd.GeoDataFrame({"geometry": cells}, crs=utm)
    grid["cell_id"] = np.arange(len(grid))
    grid.to_crs("EPSG:4326").to_file(f"{processed_dir}/grid_cells.geojson", driver="GeoJSON")
    return grid, utm

def points_to_counts(df, grid, utm_crs, processed_dir="../data/processed", freq="W"):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]), crs="EPSG:4326")
    gdf_utm = gdf.to_crs(utm_crs)

    joined = gpd.sjoin(gdf_utm[["Date","geometry"]], grid[["cell_id","geometry"]], how="inner", predicate="within")
    joined["time_dt"]  = joined["Date"].dt.tz_convert(None)
    joined["time_idx"] = joined["time_dt"].dt.to_period(freq).dt.start_time

    cell_time = (joined.groupby(["cell_id","time_idx"]).size().reset_index(name="count"))

    # full index to include zero weeks
    all_cells = grid["cell_id"].unique()
    all_times = pd.date_range(cell_time["time_idx"].min(), cell_time["time_idx"].max(), freq=freq)
    idx = pd.MultiIndex.from_product([all_cells, all_times], names=["cell_id","time_idx"])
    cell_time = (cell_time.set_index(["cell_id","time_idx"])
                        .reindex(idx, fill_value=0)
                        .reset_index())

    cell_time.to_csv(f"{processed_dir}/cell_time_counts.csv", index=False)
    return cell_time

def add_simple_features(cell_time, processed_dir="../data/processed", hotspot_frac=0.10):
    tbl = cell_time.copy()
    tbl["year"]       = tbl["time_idx"].dt.year
    tbl["weekofyear"] = tbl["time_idx"].dt.isocalendar().week.astype(int)
    tbl["month"]      = tbl["time_idx"].dt.month

    tbl = tbl.sort_values(["cell_id","time_idx"])
    for k in [1,2,3,4]:
        tbl[f"lag_{k}"] = tbl.groupby("cell_id")["count"].shift(k)

    tbl["y_count"] = tbl["count"]

    def _label_week(df_week, frac=hotspot_frac):
        if df_week["count"].sum() == 0:
            df_week["is_hot"] = 0
            return df_week
        cutoff = np.quantile(df_week["count"], 1 - frac)
        df_week["is_hot"] = (df_week["count"] >= cutoff).astype(int)
        return df_week

    tbl = tbl.groupby("time_idx", group_keys=False).apply(_label_week)
    tbl = tbl.dropna(subset=[f"lag_{k}" for k in [1,2,3,4]])
    out = f"{processed_dir}/training_table.csv"
    tbl.to_csv(out, index=False)
    return tbl

if __name__ == "__main__":
    raw_csv = "../data/raw/Crime_Data_from_2020_to_Present.csv"
    processed = "../data/processed"
    df = load_and_clean(raw_csv, processed)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]), crs="EPSG:4326")
    grid, utm = build_grid(gdf, grid_size_m=500, processed_dir=processed)
    counts = points_to_counts(df, grid, utm, processed_dir=processed, freq="W")
    tbl = add_simple_features(counts, processed_dir=processed, hotspot_frac=0.10)
    print("✅ Phase 0 complete.")