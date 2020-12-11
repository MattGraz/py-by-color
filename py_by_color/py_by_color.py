import json
from collections import Counter
from itertools import permutations
from typing import Dict

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from PIL.ImageFilter import SMOOTH, SMOOTH_MORE
from rasterio.features import shapes
from scipy.spatial.distance import euclidean
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

# AVAILABLE_COLORS = {
#      1: [228, 23, 23],  # Red-ish,
#      2: [235, 96, 96],  # Lighter Red
#      3: [230, 125, 21], # orange
#      3: [161, 92, 24], # dark orange
#      3: [100, 100, 100], # Yellow
#      3: [254,247,166], # Light Yellow
#      4: [203, 108, 0],  # light brown
#      5: [95, 136, 232], # light blue
#      5: [202, 230, 242], # light blue
#      6: [9, 56, 166],  # dark blue
#      7: [238, 127, 230],  # pink
#      8: [0, 128, 0],  # green
#      8: [5, 77, 9],  # dark green
#      8: [116,213,105],  # light green
#      9: [255, 255, 255],  # white
#      10: [0, 0, 0],  # black
# }

# IDEAS
# Color combinations
#   To understand color combos, take the weighted average of all color combos
#   This quickly adds up to WAY too many colors, even just looking at pairs
#   Is there a quick way we can estimate input a "target color" and determine how much of each
#     other color is needed to arrive at some similar amount of the other color?
# Transform color pallets
#   This isn't as easy as just changing available; Need to transform all pixels equally in some capacity


def open_image(path: str) -> Image:
    image = Image.open(path)
    return image


# image.show()
# image.info


def get_available_colors(image: Image, n_colors=15):
    pixels = list(image.getdata())
    pixel_colors_counts = Counter(pixels)

    # Cluster (Kmeans) pixels to identify "mean" color of each cluster
    km = KMeans(n_clusters=n_colors, verbose=1, random_state=100, n_jobs=8)

    km_fit = km.fit(pixels)

    colors = km_fit.cluster_centers_

    color_index_mapping = {}
    for i, color in enumerate(colors):
        color_index_mapping[i] = list(np.round(color).astype(int))

    return color_index_mapping


def map_real_colors_to_available_colors(image: Image, available_colors: Dict) -> Dict:
    # 0: R, 1: G, 2: B
    pixels = list(image.getdata())
    unique_pixel_colors = set(pixels)

    # For each pixel, determine the most similar available colors
    pixel_to_color_dict = {}
    for pixel_iter in tqdm(unique_pixel_colors):
        distances = []
        for avail_iter in available_colors.values():
            distances.append(euclidean(pixel_iter, avail_iter,))
        min_idx = np.where(distances == np.amin(distances))[0][0]
        pixel_to_color_dict[pixel_iter] = list(available_colors.values())[min_idx]

    return pixel_to_color_dict


def convert_image_to_available_colors(image: Image, color_mapping: Dict) -> Image:
    # Replace all actual pixel values with the closest available color
    image_adj = image.copy()
    width = image_adj.size[0]
    height = image_adj.size[1]
    for i in range(0, width):  # process all pixels
        for j in range(0, height):
            data = image_adj.getpixel((i, j))
            image_adj.putpixel((i, j), tuple(color_mapping[data]))

    return image_adj


def get_neighbors_idx(i, j):

    list(permutations([-1, 0, 1], 2))

    idx_adj = list(permutations([-1, 0, 1], 2))
    idx_adj.extend([[1, 1], [-1, -1]])
    neighbor_idx = []

    for i_adj, j_adj in idx_adj:

        if ((i + i_adj) < 0) or ((j + j_adj) < 0):
            continue
        else:
            neighbor_idx.append((i + i_adj, j + j_adj))

    return neighbor_idx


def get_neighbor_values(image_array: np.array, i, j):

    neighbor_idx = get_neighbors_idx(i, j)
    values = {}
    for i_neb, j_neb in neighbor_idx:
        try:
            values[(i_neb, j_neb)] = image_array[i_neb, j_neb]
        except IndexError:
            continue
    return values


def remove_single_pixels(image: Image) -> Image:
    image_adj = image.copy()
    image_adj_array = np.asarray(image_adj)
    width = image_adj_array.shape[0]
    height = image_adj_array.shape[1]
    for i in tqdm(range(0, width)):  # process all pixels
        for j in range(0, height):

            current_pixel_value = image_adj_array[i][j]

            neb_dict = get_neighbor_values(image_adj_array, i, j)
            # get value of all pixels in 9X9 box
            # neb_dict.update(get_neighbor_values(image_adj_array, i + 1, j))
            # neb_dict.update(get_neighbor_values(image_adj_array, i, j + 1))
            # neb_dict.update(get_neighbor_values(image_adj_array, i - 1, j))
            # neb_dict.update(get_neighbor_values(image_adj_array, i, j - 1))
            neb_values = [tuple(i) for i in list(neb_dict.values())]

            if tuple(current_pixel_value) not in neb_values:
                print(f"Replacing pixex {i} {j}")
                impute_pixel = Counter(neb_values).most_common()[0][0]

                # Note that (j, i) is intentional as opposed to i, j as pillow as a different ordering convetion than numpy
                image_adj.putpixel((j, i), impute_pixel)

    return image_adj


def smooth_small_polygons(image: Image, min_pixels: int = 10):
    image_adj
    # For all pixels, remove groups of pixels that are not of some minimum number of pixels
    # Impute these groups with most prevelant local color

    # This function needs to ...
    #   (1) Identify groups of pixels that are in groups less than min_pixels
    #   (2) Imputes these pixels with the most prevelant local color
    #           - Need some filter to identify the most preveland local color


def convert_image_to_shapes(image: Image, available_colors: Dict) -> gpd.GeoDataFrame:

    # Image to array
    image_adj_array = np.asarray(image)

    # Create a mask for each available color in the image, convert each to multipolygons
    keep_shapes = []
    for color_id_for_mask in tqdm(available_colors.items()):
        color_for_mask = available_colors[color_id_for_mask[0]]
        image_raster_adj_mask = np.all(image_adj_array == color_for_mask, axis=2)
        all_shapes_color = shapes(image_raster_adj_mask.astype("int32"))

        for shape_iter in all_shapes_color:
            # If shape has value 1 (True), keep it else drop
            if shape_iter[1] == 1:
                geo = gpd.GeoDataFrame(
                    geometry=[Polygon(s) for s in shape_iter[0]["coordinates"]]
                )
                geo["color_name"] = str(color_for_mask)

                # Set value to color
                # shape_iter[1] = "Color_Placeholder"
                # keep_shapes.append(shape_iter[0])
                keep_shapes.append(geo)

    all_geo = pd.concat(keep_shapes)
    return all_geo


def smooth_image(image: image, iter=3) -> image:
    for _ in range(0, iter):
        image = image.filter(SMOOTH_MORE)
    return image


def main():
    image = open_image("data/yosemite_small.png")
    available_colors = get_available_colors(image, 15)
    image = smooth_image(
        image, 2
    )  # Helpful when there are very fine details; Can I programatically identify?
    pixel_to_color_dict = map_real_colors_to_available_colors(image, available_colors)
    image_adj = convert_image_to_available_colors(image, pixel_to_color_dict)
    image_adj_new = remove_single_pixels(image_adj)
    image_adj_gdf = convert_image_to_shapes(image_adj_new, available_colors)
    image_adj_gdf.to_file("test_geopandas.geojson", driver="GeoJSON")
    # image_adj_gdf.plot()
    # plt.show()


if __name__ == "__main__":
    main()
