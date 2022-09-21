import pandas as pd
import numpy as np
import os

# https://stackoverflow.com/questions/53233228/plot-latitude-longitude-from-csv-in-python-3-6
# https://geopandas.org/en/latest/gallery/cartopy_convert.html
# https://stackoverflow.com/questions/55646598/polar-stereographic-projection-of-geopandas-world-map
# https://geopandas.org/en/latest/gallery/cartopy_convert.html
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from shapely.geometry import Point, Polygon, MultiPoint
import geopandas as gpd
import cartopy.crs as ccrs
from shapely.ops import triangulate, voronoi_diagram

m_paths = os.path.join('data', 'srs_csv_files')
x_path = os.path.join('data', 'data_xe133_NH_yearly.csv')
y_path = os.path.join('data', 'scalings.csv')

scalings = pd.read_csv(y_path, header = None)
sources = pd.read_csv(x_path)
sources['scaled'] = sources['emission'] * 1000 / scalings[0]

# remove duplicate days
samples = pd.DataFrame([f.split('_') for f in os.listdir(m_paths)],
                       columns = ['obs', 'd', 't'])
samples.drop_duplicates(subset = ['obs', 'd'], inplace = True)

# keep days with all stations
days = samples.groupby('d')['obs'].apply(set)
s, c = np.unique(days.values, return_counts = True)
print(c.max(), s[c.argmax()])
best = days[days == s[c.argmax()]]
samples2 = samples[samples['d'].isin(best.index.values)].copy()

## todo, most consecutive

for row in samples2.iterrows():
    f = '_'.join(samples2.iloc[0])
    data = pd.read_csv(os.path.join(m_paths, f), header = None)
    # np.matmul(data.T, sources['scaled'].values)
    z = data.sum(axis = 1) # total over 15 days

    zm = z.max() - 2 * z.median()
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin = z.median() - 2 * zm, vmax = z.median() + 2 * zm)
    cmap = cm.hot
    m = cm.ScalarMappable(norm = norm, cmap = cmap)
    colos = m.to_rgba(z)
    
    ## TODO: sum influence from all stations
    ## todo: fix less polygons 196 than stations 200

    ## view
    crs = ccrs.NorthPolarStereo()
    crs_proj4 = crs.proj4_init

    ## patches
    points = MultiPoint([v for v in zip(sources['Longitude'], sources['Latitude'])])
    regions = voronoi_diagram(points)
    polygons = [v for v in regions.geoms]
    gdf = gpd.GeoDataFrame(index = range(len(polygons)), geometry = polygons, crs = 'EPSG:4326') # EPSG 4326 - WGS 84
    gdf2 = gdf.to_crs(crs_proj4) # project points to new CRS

    ## markers
    geometry = [Point(xy) for xy in zip(sources['Longitude'], sources['Latitude'])]
    gdf3 = gpd.GeoDataFrame(sources, geometry = geometry, crs = 'EPSG:4326') # EPSG 4326 - WGS 84
    gdf4 = gdf3.to_crs(crs_proj4) # project points to new CRS


    ## background
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world2 = world.to_crs(crs_proj4)

    ## plot
    fig, ax = plt.subplots(subplot_kw = {'projection': crs}, figsize = (8, 6))
    ax.add_geometries(world2['geometry'], crs = crs)
    ##gdf2.plot(ax = ax, marker = 'o', color = 'red', alpha = 0.5, zorder = 10)
    gdf2.plot(ax = ax, marker = 'o', color = colos, alpha = 0.5, zorder = 10)
    gdf4.plot(ax = ax, marker = 'o', color = 'red', markersize = 5, zorder = 11)
    plt.show()

    #ax2 = world.plot(figsize = (8, 6))
    #gdf.plot(ax = ax2, marker = 'o', color = colos, alpha = 0.5, zorder = 10)
    #gdf3.plot(ax = ax2, marker = 'o', color = 'red', markersize = 5)
    #plt.show()

    x

    ## TODO: animation
