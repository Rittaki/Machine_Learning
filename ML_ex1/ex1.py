import matplotlib.pyplot as plt
import numpy as np
import sys

image_fname, centroids_fname, out_fname = sys.argv[1], sys.argv[2], sys.argv[3]
f = open(out_fname, "a")
z = np.loadtxt(centroids_fname) #load centroids
number_of_clusters = z.shape[0]
collection_of_clusters = dict()

for i in range(number_of_clusters):
    collection_of_clusters["cluster" + str(i)] = []

orig_pixels = plt.imread(image_fname)
pixels = orig_pixels.astype(float)/255.
# Reshape the image(128x128x3) into  an Nx3 matrix where N = number of pixels
pixels = pixels.reshape(-1, 3)
distances = []

# for i, pixel in enumerate(pixels):
#

for i, center in enumerate(z):
    dist_for_single_centroid = np.linalg.norm(center - pixels, axis=1)
    distances.append(dist_for_single_centroid)
    print("dist: ", dist_for_single_centroid)
    print("min dist ", np.min(dist_for_single_centroid))

    # Add the number of the pixel-id (0-16384)
    # collection_of_clusters["cluster" + str(i)].append([12121454545])
# for i, array in enumerate(distances):
#     for j in array:
#         print(distances[i][0])
count = 0
for i in distances[0]:
    if distances[0][count] < distances[1][count]:
        collection_of_clusters["cluster" + str(0)].append(pixels[count])
    else:
        collection_of_clusters["cluster" + str(1)].append(pixels[count])
    # print(distances[0][count])
    count = count + 1

new_centroids = []
np_collection_of_clusters_first = np.array(collection_of_clusters["cluster" + str(0)])
avg_first = np.array(np.average(np_collection_of_clusters_first.reshape(-1, 3), axis=0).round(4))
np_collection_of_clusters_second = np.array(collection_of_clusters["cluster" + str(1)])
avg_second = np.array(np.average(np_collection_of_clusters_second.reshape(-1, 3), axis=0).round(4))
new_centroids.append(avg_first)
new_centroids.append(avg_second)
print(avg_first)
print(avg_second)

f.write(f"[iter {0}]:{','.join([str(i) for i in new_centroids])}")

exit()