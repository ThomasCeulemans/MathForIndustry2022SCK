import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mplc
from shapely.geometry import Point, Polygon, MultiPoint
import geopandas as gpd
import cartopy.crs as ccrs
from shapely.ops import triangulate, voronoi_diagram

basedir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
i_paths = os.path.join(basedir, 'input')
o_paths = os.path.join(basedir, 'output')

x_path = os.path.join(basedir, 'data', 'data_xe133_NH_yearly.csv')
y_path = os.path.join(basedir, 'data', 'scalings.csv')

scalings = pd.read_csv(y_path, header = None)
sources = pd.read_csv(x_path)
sources['scaled'] = sources['emission'] * 1000 / scalings[0]

day0 = '20140101'
daymin = pd.to_datetime(day0) + pd.DateOffset(-15)
daymax = pd.to_datetime(day0) + pd.DateOffset(364)
dates = pd.date_range(daymin, daymax)


for f in os.listdir(i_paths):

    # file and folder names
    name = f[:-4]
    key = name.split('_')[0]
    lamb = name.split('_')[3]
    name = f[:-4]
    dir1 = os.path.join(o_paths, '{}{}_pole'.format(key, lamb))
    dir2 = os.path.join(o_paths, '{}{}_map'.format(key, lamb))
    os.makedirs(dir1, exist_ok = True)
    os.makedirs(dir2, exist_ok = True)

    # fix input data shape
    data = pd.read_csv(os.path.join(i_paths, f), header = None).values.reshape((200, 380))
    df = pd.DataFrame({k: v for k, v in zip(sources['Facility'], data)}, index = dates)
    df.to_csv(os.path.join(o_paths, f))

    # calc plot limits
    p = 1
    zmin = np.percentile(data, p)
    zmax = np.percentile(data, 100 - p)

    for d, obs in df.iterrows():
        day = d.strftime('%Y-%m-%d')
        print(name, d, key, lamb)

        # order is preserved, no need to lookup
        z = obs.values

        ## view
        crs = ccrs.NorthPolarStereo()
        crs_proj4 = crs.proj4_init

        ## markers
        geometry = [Point(xy) for xy in zip(sources['Longitude'], sources['Latitude'])]
        gdf3 = gpd.GeoDataFrame(sources, geometry = geometry, crs = 'EPSG:4326') # EPSG 4326 - WGS 84
        gdf4 = gdf3.to_crs(crs_proj4) # project points to new CRS

        ## patches
        points = MultiPoint([v for v in zip(sources['Longitude'], sources['Latitude'])])
        regions = voronoi_diagram(points)
        polygons = [v for v in regions.geoms]
        gdf1 = gpd.GeoDataFrame(index = range(len(polygons)), geometry = polygons, crs = 'EPSG:4326') # EPSG 4326 - WGS 84
        #gdf2 = gdf1.to_crs(crs_proj4) # project points to new CRS
        points2 = MultiPoint([v for v in gdf4['geometry']])
        regions2 = voronoi_diagram(points2)
        polygons2 = [v for v in regions2.geoms]
        gdf2 = gpd.GeoDataFrame(index = range(len(polygons2)), geometry = polygons2, crs = crs)

        ## background
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        world2 = world.to_crs(crs_proj4)


        ## add values for overlaping points
        # todo: group very close regions
        poly_pts = [[k for k, pt in enumerate(geometry) if pol.contains(pt)] for pol in polygons]
        poly_colors = [z[idx].sum() for idx in poly_pts]

        # color map
        norm = mplc.Normalize(vmin = zmin, vmax = zmax)
        cmap = cm.jet
        m = cm.ScalarMappable(norm = norm, cmap = cmap)
        colos = m.to_rgba(z)

        ## plots
        title = r'{}      {}      $\lambda$ = {}'.format(day, key, lamb)

        # north
        fig, ax = plt.subplots(subplot_kw = {'projection': crs}, figsize = (20, 20))
        ax.add_geometries(world2['geometry'], crs = crs)
        gdf2.plot(ax = ax, marker = 'o', color = colos, alpha = 0.8, zorder = 10)
        gdf4.plot(ax = ax, marker = 'o', color = 'red', markersize = 5, zorder = 11)

        minx, miny, maxx, maxy = gdf4.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        ax.set_title(title)
        ax.figure.colorbar(m, ax = ax, fraction=0.02, pad=0.02)
        plt.savefig(os.path.join(dir1, '{}.png'.format(day)), bbox_inches = 'tight')
        plt.close(fig)

        # normal
        ax2 = world.plot(figsize = (20, 20))
        gdf1.plot(ax = ax2, marker = 'o', color = colos, alpha = 0.8, zorder = 10)
        gdf3.plot(ax = ax2, marker = 'o', color = 'red', markersize = 5, zorder = 11)

        minx, miny, maxx, maxy = world.total_bounds
        ax2.set_xlim(minx, maxx)
        ax2.set_ylim(0, maxy)

        ax2.set_title(title)
        ax2.figure.colorbar(m, ax = ax2, fraction=0.015, pad=0.02)
        plt.savefig(os.path.join(dir2, '{}.png'.format(day)), bbox_inches = 'tight')
        plt.close(ax2.figure)


# https://stackoverflow.com/questions/53233228/plot-latitude-longitude-from-csv-in-python-3-6
# https://geopandas.org/en/latest/gallery/cartopy_convert.html
# https://stackoverflow.com/questions/55646598/polar-stereographic-projection-of-geopandas-world-map

## TODO
# animation
# https://underworldcode.github.io/stripy/2.0.5b2/FrontPage.html
