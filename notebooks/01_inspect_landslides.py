import geopandas as gpd

file_path = "data/raw/sikkim_landslides/Google_Earth_landslides_point_21Dec2021.shp"

gdf = gpd.read_file(file_path)

print("Number of landslides:", len(gdf))
print("\nColumns:")
print(gdf.columns.tolist())

print("\nFirst 5 records:")
print(gdf.head())

print("\nCoordinate system:")
print(gdf.crs)

print("\nMissing values:")
print(gdf.isnull().sum())