import matplotlib.pyplot as plt
import numpy as np
import sys

image_fname, centroids_fname, out_fname = sys.argv[1], sys.argv[2], sys.argv[3]
f = open(out_fname, "a")
z = np.loadtxt(centroids_fname) #load centroids
number_of_clusters = z.shape[0]
collection_of_clusters = dict()

# for i in range(number_of_clusters):
#     collection_of_clusters["cluster" + str(i)] = []

orig_pixels = plt.imread(image_fname)
pixels = orig_pixels.astype(float)/255.
# Reshape the image(128x128x3) into  an Nx3 matrix where N = number of pixels
pixels = pixels.reshape(-1, 3)
# for i, pixel in enumerate(pixels):
#

# new_centroids = []
old_centroids = z
iterations = 0
while True:
    if iterations == 20:
        break
    distances = []
    for i in range(number_of_clusters):
        collection_of_clusters["cluster" + str(i)] = []

    for i, center in enumerate(old_centroids):
        dist_for_single_centroid = np.linalg.norm(center - pixels, axis=1)
        distances.append(dist_for_single_centroid)
        print("dist: ", dist_for_single_centroid)
        print("min dist ", np.min(dist_for_single_centroid))

    count = 0
    len_of_dists = len(distances[0])
    # for i in range(len_of_dists):
    #
    array_of_mins = np.min(distances, axis=0)
    for i in range(len_of_dists):
        for j in range(number_of_clusters):
            if distances[j][i] == array_of_mins[i]:
                collection_of_clusters["cluster" + str(j)].append(pixels[i])
    # for i in distances[0]:
    #     if distances[0][count] < distances[1][count]:
    #         collection_of_clusters["cluster" + str(0)].append(pixels[count])
    #     else:
    #         collection_of_clusters["cluster" + str(1)].append(pixels[count])
    #     # print(distances[0][count])
    #     count = count + 1
    new_centroids = []
    for i in range(number_of_clusters):
        np_collection_of_clusters = np.array(collection_of_clusters["cluster" + str(i)])
        avg = np.array(np.average(np_collection_of_clusters.reshape(-1, 3), axis=0).round(4))
        # new_centroids = []
        new_centroids.append(avg)

    new_centroids = np.array(new_centroids)

    # np_collection_of_clusters_first = np.array(collection_of_clusters["cluster" + str(0)])
    # np_collection_of_clusters_second = np.array(collection_of_clusters["cluster" + str(1)])
    # avg_first = np.array(np.average(np_collection_of_clusters_first.reshape(-1, 3), axis=0).round(4))
    # avg_second = np.array(np.average(np_collection_of_clusters_second.reshape(-1, 3), axis=0).round(4))
    # new_centroids = np.vstack([avg_first, avg_second])
    #
    # stam_centroids = []
    # stam_centroids.append(avg_first)
    # stam_centroids.append(avg_second)
    # stam_centroids = np.array(stam_centroids)
    # print(avg_first)
    # print(avg_second)

    f.write(f"[iter {iterations}]:{','.join([str(i) for i in new_centroids])}")
    f.write("\n")
    if np.array_equal(new_centroids, old_centroids):
        break
    old_centroids = new_centroids
    iterations = iterations + 1


    #
    # if iterations == 20:
    #     break
    # old_centroids = z
    # for i, center in enumerate(z):
    #     dist_for_single_centroid = np.linalg.norm(center - pixels, axis=1)
    #     distances.append(dist_for_single_centroid)
    #     print("dist: ", dist_for_single_centroid)
    #     print("min dist ", np.min(dist_for_single_centroid))
    #
    # count = 0
    # for i in distances[0]:
    #     if distances[0][count] < distances[1][count]:
    #         collection_of_clusters["cluster" + str(0)].append(pixels[count])
    #     else:
    #         collection_of_clusters["cluster" + str(1)].append(pixels[count])
    #     # print(distances[0][count])
    #     count = count + 1
    #
    # np_collection_of_clusters_first = np.array(collection_of_clusters["cluster" + str(0)])
    #
    # avg_first = np.array(np.average(np_collection_of_clusters_first.reshape(-1, 3), axis=0).round(4))
    # np_collection_of_clusters_second = np.array(collection_of_clusters["cluster" + str(1)])
    # avg_second = np.array(np.average(np_collection_of_clusters_second.reshape(-1, 3), axis=0).round(4))
    # new_centroids = np.vstack([avg_first, avg_second])
    #
    # print(avg_first)
    # print(avg_second)
    #
    # f.write(f"[iter {iterations}]:{','.join([str(i) for i in new_centroids])}")
    # f.write("\n")
    # iterations = iterations + 1