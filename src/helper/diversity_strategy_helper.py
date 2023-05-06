import numpy as np
import torch

def get_indices_of_2_smallest(arr):
    k = 2
    idx = np.argpartition(arr.ravel(), k)
    nested = np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))]
    idx_smallest = np.array([nested[0][0], nested[0][1]])
    #values_smallest = arr[idx_smallest]
    return idx_smallest


def check_cluster_sizes(margins, number_clusters, num_of_boundaries_per_cluster):
    clusters_too_small = False
    j = 0
    while j < number_clusters:
        amount_indices_current_cluster = len(np.where(margins[:, 0] == j)[0])
        if amount_indices_current_cluster < num_of_boundaries_per_cluster:
            clusters_too_small = True
        j += 1
    return clusters_too_small

def compute_boundary_points(distances, k):
    amount_points = len(distances[:, 0])  # len(distances[:,0])=10_000
    i = 0
    idx_centroids = get_indices_of_2_smallest(distances[i, :])
    distances_to_centroids = distances[i][idx_centroids]
    margin = distances_to_centroids[1] - distances_to_centroids[0]
    margins = np.array(margin)
    i += 1
    while i < amount_points:
        idx_centroids = get_indices_of_2_smallest(distances[i, :])
        distances_to_centroids = distances[i][idx_centroids]
        current_margin = distances_to_centroids[1] - distances_to_centroids[0]
        margins = np.append(margins, current_margin)
        i += 1
    margin_tensor = torch.from_numpy(margins)
    values, smallest_margin_indices = torch.topk(margin_tensor, k, largest=False)
    return smallest_margin_indices

def compute_boundary_points_overflow_canceled(distances, diversity_strategy, k, number_clusters):
    if diversity_strategy == "bp_smo_canceled":
        isLargest = False
    if diversity_strategy == "bp_lmo_canceled":
        isLargest = True
    amount_points = len(distances[:, 0])  # len(distances[:,0])=10_000
    i = 0
    idx_centroids = get_indices_of_2_smallest(distances[i, :])
    distances_to_centroids = distances[i][idx_centroids]
    margin = distances_to_centroids[1] - distances_to_centroids[0]
    centroid = np.array(idx_centroids[0])
    margins = np.array([centroid, margin])
    num_of_boundaries_per_cluster = int(k/ number_clusters)
    i += 1
    while i < amount_points:
        idx_centroids = get_indices_of_2_smallest(distances[i, :])
        distances_to_centroids = distances[i][idx_centroids]
        current_margin = distances_to_centroids[1] - distances_to_centroids[0]
        current_centroid = np.array(idx_centroids[0])
        current_double = np.array([current_centroid, current_margin])
        margins = np.vstack((margins, current_double))
        i += 1
    clusters_too_small = False
    p = 0
    while p < number_clusters:
        amount_indices_current_cluster = len(np.where(margins[:, 0] == p)[0])
        if amount_indices_current_cluster < num_of_boundaries_per_cluster:
            clusters_too_small = True
        p += 1
    i += 1
    if clusters_too_small == False:
        print("clusters big enough")
        j = 0
        current_cluster_indices = np.where(margins[:, 0] == j)
        current_cluster_values = margins[current_cluster_indices][:, 1]
        current_margin_tensor = torch.from_numpy(current_cluster_values)
        values, indices = torch.topk(current_margin_tensor, num_of_boundaries_per_cluster, largest=isLargest)
        smallest_margin_indices = torch.from_numpy(current_cluster_indices[0])[indices]
        j += 1
        while j < number_clusters:
            current_cluster_indices = np.where(margins[:, 0] == j)
            current_cluster_values = margins[current_cluster_indices][:, 1]
            current_margin_tensor = torch.from_numpy(current_cluster_values)
            values, indices = torch.topk(current_margin_tensor, num_of_boundaries_per_cluster,
                                         largest=isLargest)
            current_cluster_margin_indices = torch.from_numpy(current_cluster_indices[0])[indices]
            smallest_margin_indices = torch.cat([smallest_margin_indices, current_cluster_margin_indices], dim=0)
            j += 1
    else:
        print("clusters too small")
        current_cluster_values = margins[:, 1]
        current_margin_tensor = torch.from_numpy(current_cluster_values)
        values, smallest_margin_indices = torch.topk(current_margin_tensor, k, largest=isLargest)
    return smallest_margin_indices

def get_portions_depending_on_clustersize(margins, k, number_clusters, total_amount_points):
    # berechnet wie groß der Anteil pro Cluster ist (immer abgerundet)
    j = 0
    amount_points_current_cluster = len(np.where(margins[:, 0] == j)[0])
    percentage = amount_points_current_cluster / total_amount_points
    bp_per_cluster = int(percentage * k)
    bp_per_cluster_array = np.array(int(percentage * k))
    points_total_so_far = bp_per_cluster
    idx_cluster = j
    isScaledDown = False
    if bp_per_cluster < round(percentage * k):
        isScaledDown = True
    triplet = np.array([idx_cluster, isScaledDown, bp_per_cluster])
    j += 1
    while j < number_clusters:
        amount_points_current_cluster = len(np.where(margins[:, 0] == j)[0])
        percentage = amount_points_current_cluster/total_amount_points
        bp_per_cluster = int(percentage*k)
        points_total_so_far += bp_per_cluster
        idx_cluster = j
        isScaledDown = False
        if bp_per_cluster<(percentage*k):
            isScaledDown = True
        current_triplet = np.array([idx_cluster, isScaledDown, bp_per_cluster])
        bp_per_cluster_array = np.append(bp_per_cluster_array, bp_per_cluster)
        triplet = np.vstack((triplet, current_triplet))
        j += 1
    # Falls es zu klein ist, dann die größten Cluster, die abgerundet wurden suchen
    if points_total_so_far < k:
        i = 0
        indices_isScaledDown = 0
        while i < number_clusters:
            isScaledDown = triplet[i][1]
            if (isScaledDown):
                idx = triplet[i][0] # idx des clusters
                current_idx_isScaledDown = np.array(idx) # idx des clusters als np array
                indices_isScaledDown = np.append(indices_isScaledDown, current_idx_isScaledDown)
            i += 1
        indices_isScaledDown = np.delete(indices_isScaledDown, 0)
        current_cluster_proportions = triplet[indices_isScaledDown][:,2]
        current_cluster_proportions_tensor = torch.from_numpy(current_cluster_proportions)
        values, indices = torch.topk(current_cluster_proportions_tensor, (k - points_total_so_far), largest=True)
        correct_indices = indices_isScaledDown[indices]
        for idx_isScaledDown in correct_indices:
            bp_per_cluster_array[idx_isScaledDown] = bp_per_cluster_array[idx_isScaledDown]+1
    return bp_per_cluster_array

def compute_boundary_prop_clustersize(distances, k, number_clusters):
    amount_points = len(distances[:, 0])  # len(distances[:,0])=10_000
    i = 0
    idx_centroids = get_indices_of_2_smallest(distances[i, :])
    distances_to_centroids = distances[i][idx_centroids]
    margin = distances_to_centroids[1] - distances_to_centroids[0]
    centroid = np.array(idx_centroids[0])
    margins = np.array([centroid, margin])
    i += 1
    while i < amount_points:
        idx_centroids = get_indices_of_2_smallest(distances[i, :])
        distances_to_centroids = distances[i][idx_centroids]
        current_margin = distances_to_centroids[1] - distances_to_centroids[0]
        current_centroid = np.array(idx_centroids[0])
        current_double = np.array([current_centroid, current_margin])
        margins = np.vstack((margins, current_double))
        i += 1
    j = 0

    boundary_points_per_cluster = get_portions_depending_on_clustersize(margins, k, number_clusters, amount_points)
    current_cluster_indices = np.where(margins[:, 0] == j)
    current_cluster_values = margins[current_cluster_indices][:, 1]

    current_margin_tensor = torch.from_numpy(current_cluster_values)
    values, indices = torch.topk(current_margin_tensor, boundary_points_per_cluster[j], largest=False)
    smallest_margin_indices = torch.from_numpy(current_cluster_indices[0])[indices]
    j += 1
    while j < number_clusters:
        current_cluster_indices = np.where(margins[:, 0] == j)
        current_cluster_values = margins[current_cluster_indices][:, 1]
        current_margin_tensor = torch.from_numpy(current_cluster_values)
        if(boundary_points_per_cluster[j]>0):
            values, indices = torch.topk(current_margin_tensor, boundary_points_per_cluster[j],
                                         largest=False)
            current_cluster_margin_indices = torch.from_numpy(current_cluster_indices[0])[indices]
            smallest_margin_indices = torch.cat([smallest_margin_indices, current_cluster_margin_indices], dim=0)
        j += 1
    return smallest_margin_indices

