import numpy as np
import torch

foo = np.array([
    [4,888,6,885,930,33],
    [3,4,5,6,1,2],
    [9,34,53,2,77,88]])

def get_indices_of_2_smallest(arr):
    k = 2
    idx = np.argpartition(arr.ravel(), k)
    nested = np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))]
    idx_smallest = np.array([nested[0][0], nested[0][1]])
    values_smallest = arr[idx_smallest]
    return idx_smallest

def compute_boundary_points(distances):
    amount_points = len(distances[:,0]) # len(distances[:,0])=10_000
    i = 0
    idx_centroids = get_indices_of_2_smallest(distances[i,:])
    distances_to_centroids = distances[i][idx_centroids]
    margin = distances_to_centroids[1] - distances_to_centroids[0]
    margins = np.array(margin)
    i+=1
    while i<amount_points:
        print("i")
        print(i)
        #suche 2 kleinsten bstände zum centroid für alle 10_000:
        #     abspeichern idx_point_10_000, idx_centroids_1_000, distances_to_centroids, label_point
        idx_centroids = get_indices_of_2_smallest(distances[i,:])
        distances_to_centroids = distances[i][idx_centroids]
        #berechne margin der 2 kleinsten:
        #     abspeichern idx_point_10_000, margin_value in einem array
        current_margin = distances_to_centroids[1] - distances_to_centroids[0]
        # margin & idx abspeichern:
        margins = np.append(margins, current_margin)
        i += 1
    margin_tensor = torch.from_numpy(margins)
    values, smallest_margin_indices = torch.topk(margin_tensor, 2, largest=False)
    #array durchlaufen und 1000 kleinsten margins mit idx verwenden
    return smallest_margin_indices

compute_boundary_points(foo)

def get_portions_depending_on_clustersize(margins, settings, total_amount_points):
    # berechnet wie groß der Anteil pro Cluster ist (immer abgerundet)
    j = 0
    points_total_so_far = 0
    amount_points_current_cluster = len(np.where(margins[:, 0] == j)[0])
    percentage = amount_points_current_cluster / total_amount_points
    bp_per_cluster = int(percentage * settings["k"])
    bp_per_cluster_array = np.array(int(percentage * settings["k"]))
    points_total_so_far += points_total_so_far
    idx_cluster = j
    isScaledDown = False
    if bp_per_cluster < round(percentage * settings["k"]):
        isScaledDown = True
    triplet = np.array([idx_cluster, isScaledDown, bp_per_cluster])
    j += 1
    while j < settings["number_clusters"]:
        amount_points_current_cluster = len(np.where(margins[:, 0] == j)[0])
        percentage = amount_points_current_cluster/total_amount_points
        bp_per_cluster = int(percentage*settings["k"])
        points_total_so_far += points_total_so_far
        idx_cluster = j
        isScaledDown = False
        if bp_per_cluster<round(percentage*settings["k"]):
            isScaledDown = True
        current_triplet = np.array([idx_cluster, isScaledDown, bp_per_cluster])
        bp_per_cluster_array = np.append(bp_per_cluster_array, bp_per_cluster)
        triplet = np.vstack((triplet, current_triplet))
        j += 1
    # Falls es zu klein ist, dann die größten Cluster, die abgerundet wurden suchen
    if points_total_so_far < settings["k"]:
        i = 0
        isScaledDown = triplet[i][1]
        if (isScaledDown):
            idx = triplet[i][0]
            indices_isScaledDown = np.array(idx)
        i += 1
        while i < settings["number_clusters"]:
            isScaledDown = triplet[i][1]
            if (isScaledDown):
                idx = triplet[i][0]
                current_idx_isScaledDown = np.array(idx)
                indices_isScaledDown = np.append(indices_isScaledDown, current_idx_isScaledDown)
            i += 1
        current_cluster_proportions = triplet[indices_isScaledDown][:,2]
        current_cluster_proportions_tensor = torch.from_numpy(current_cluster_proportions)
        values, indices = torch.topk(current_cluster_proportions_tensor, (settings["k"]-points_total_so_far), largest=True)
        #todo: Check if for-loop is correct
        # Zu diesen größten, abgerundeten Clustern jeweils noch einen Punkt hinzufügen
        for idx_isScaledDown in indices:
            bp_per_cluster_array[idx_isScaledDown] = bp_per_cluster_array[idx_isScaledDown]+1
    return bp_per_cluster_array

def compute_boundary_prop_clustersize(distances, settings):
    amount_points = len(distances[:, 0])  # len(distances[:,0])=10_000
    i = 0
    idx_centroids = get_indices_of_2_smallest(distances[i, :])
    distances_to_centroids = distances[i][idx_centroids]
    margin = distances_to_centroids[1] - distances_to_centroids[0]
    centroid = np.array(idx_centroids[0])
    margins = np.array([centroid, margin])
    num_of_boundaries_per_cluster = int(settings["k"] / settings["number_clusters"])
    i += 1
    print("before amount_points")
    while i < amount_points:
        idx_centroids = get_indices_of_2_smallest(distances[i, :])
        distances_to_centroids = distances[i][idx_centroids]
        current_margin = distances_to_centroids[1] - distances_to_centroids[0]
        current_centroid = np.array(idx_centroids[0])
        current_double = np.array([current_centroid, current_margin])
        margins = np.vstack((margins, current_double))
        i += 1
    j = 0
    #todo: Check in which form "boundary_points_per_cluster" is returned
    boundary_points_per_cluster = get_portions_depending_on_clustersize(margins, settings, amount_points)
    current_cluster_indices = np.where(margins[:, 0] == j)
    current_cluster_values = margins[current_cluster_indices][:, 1]
    current_margin_tensor = torch.from_numpy(current_cluster_values)
    values, indices = torch.topk(current_margin_tensor, boundary_points_per_cluster[j], largest=False)
    smallest_margin_indices = torch.from_numpy(current_cluster_indices[0])[indices]
    j += 1
    while j < settings["number_clusters"]:
        current_cluster_indices = np.where(margins[:, 0] == j)
        current_cluster_values = margins[current_cluster_indices][:, 1]
        current_margin_tensor = torch.from_numpy(current_cluster_values)
        values, indices = torch.topk(current_margin_tensor, boundary_points_per_cluster[j],
                                         largest=False)
        current_cluster_margin_indices = torch.from_numpy(current_cluster_indices[0])[indices]
        smallest_margin_indices = torch.cat([smallest_margin_indices, current_cluster_margin_indices], dim=0)
        j += 1
    return smallest_margin_indices