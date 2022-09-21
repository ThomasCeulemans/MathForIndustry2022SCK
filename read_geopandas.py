import pandas as pd
import numpy as np
import os
import itertools
from collections import Counter

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mplc
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

# keep days with most common stations
stations = samples['obs'].unique()
days = samples.groupby('d')['obs'].apply(set) # groupby days, get set of available stations

comb = [set(v) for i in range(1, len(stations) + 1) for v in itertools.combinations(stations, i)] # all possible combinations
count = [days.apply(lambda x: c.issubset(x)).sum() for c in comb] # count combination is subset
description = pd.DataFrame({'subsets': comb,
                            'len': [len(c) for c in comb],
                            'count': count})
#description.to_csv('common_subsets.csv')

# use subset with most stations
thresh = 100 # at least 100 days (*note later remove non consecutive)
availb = description[description['count'] > thresh]
availb.iloc[availb['len'].argmax()]
subset, sublen, subcnt = availb.iloc[availb['len'].argmax()]

days2 = days[days.apply(lambda x: subset.issubset(x))].copy()
samples2 = samples[samples['obs'].isin(subset) & samples['d'].isin(days2.index.values)].copy()

s, c = np.unique(samples2['d'], return_counts = True)
print(c.min() == sublen, sublen, 'stations,', subcnt, 'days') # safety check

# keep consecutive days
daysdiff = pd.to_datetime(days2.index).to_series().diff().astype('timedelta64[D]').values
breaks = np.where(daysdiff != 1)[0]
mcdidx = np.diff(breaks).argmax() # most consecutive days index
start = breaks[mcdidx]
stop = breaks[mcdidx + 1] - 1
print(days2.index[start], '>', days2.index[stop], np.diff(breaks).max(), 'consecutive days')
days3 = days2.iloc[range(start, stop + 1)].copy()

# final subset
samples3 = samples2[samples2['d'].isin(days3.index.values)].copy()
#samples3.apply('_'.join, axis = 1).to_csv('48d8sta.csv')

for day, day_stations in days3.iteritems():
    files = samples3[samples3['d'] == day].apply('_'.join, axis = 1)

    ## sum influence from all stations
    #for f in files:
    #    data = pd.read_csv(os.path.join(m_paths, f), header = None)
    #    # np.matmul(data.T, sources['scaled'].values)
    #    z = data.sum(axis = 1) # total over 15 days

    ## sum influence from all stations
    daily = []
    for f in files:
        data = pd.read_csv(os.path.join(m_paths, f), header = None)
        # np.matmul(data.T, sources['scaled'].values)
        dc = data[data > 0].sum(axis = 1) # total over 15 days, positive contributions only
        daily += [dc]
    z = np.array(daily).sum(axis = 0)

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

    ## add values for overlaping points
    # todo: group very close regions
    poly_pts = [[k for k, pt in enumerate(geometry) if pol.contains(pt)] for pol in polygons]
    poly_colors = [z[idx].sum() for idx in poly_pts]

    # color map
    #norm = mplc.Normalize(vmin = np.percentile(z, 5), vmax = np.percentile(z, 95))
    norm = mplc.Normalize(vmin = 0, vmax = 100)
    cmap = cm.jet
    m = cm.ScalarMappable(norm = norm, cmap = cmap)
    colos = m.to_rgba(z)
    
    ## plot
    fig, ax = plt.subplots(subplot_kw = {'projection': crs}, figsize = (16, 16))
    ax.add_geometries(world2['geometry'], crs = crs)
    gdf2.plot(ax = ax, marker = 'o', color = colos, alpha = 0.5, zorder = 10)
    gdf4.plot(ax = ax, marker = 'o', color = 'red', markersize = 5, zorder = 11)

    minx, miny, maxx, maxy = gdf4.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    fig.colorbar(m, ax = ax, fraction=0.02, pad=0.02)
    #plt.show()

    #ax2 = world.plot(figsize = (8, 6))
    #gdf.plot(ax = ax2, marker = 'o', color = colos, alpha = 0.5, zorder = 10)
    #gdf3.plot(ax = ax2, marker = 'o', color = 'red', markersize = 5)
    #plt.show()

    plt.savefig(os.path.join('img', '{}.png'.format(day)), bbox_inches = 'tight')
    plt.close(fig)


# https://stackoverflow.com/questions/53233228/plot-latitude-longitude-from-csv-in-python-3-6
# https://geopandas.org/en/latest/gallery/cartopy_convert.html
# https://stackoverflow.com/questions/55646598/polar-stereographic-projection-of-geopandas-world-map

## TODO
# animation
# https://underworldcode.github.io/stripy/2.0.5b2/FrontPage.html
# https://coolum001.github.io/voronoi.html


