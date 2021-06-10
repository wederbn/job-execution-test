"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

import matplotlib.pyplot as plt
import asyncio
import numpy as np
from qiskitSerializer import QiskitSerializer
from quantumBackendFactory import QuantumBackendFactory
from clusteringCircuitGenerator import ClusteringCircuitGenerator
from dataProcessingService import DataProcessingService
from negativeRotationClusteringService import NegativeRotationClusteringService
from convergenceCalculationService import ConvergenceCalculationService
from fileService import FileService
from shutil import copyfile


async def initialize_centroids(job_id, k):
    """
    Create k random centroids in the range [0, 1] x [0, 1].
    """

    centroids_file_path = './static/centroid-calculation/initialization/centroids' \
                          + str(job_id) + '.txt'

    # create working folder if not exist
    FileService.create_folder_if_not_exist('./static/centroid-calculation/initialization/')

    # delete old files if exist
    FileService.delete_if_exist(centroids_file_path)

    # generate k centroids
    centroids = DataProcessingService.generate_random_data(k)

    # serialize the data
    np.savetxt(centroids_file_path, centroids)

    return centroids_file_path


async def calculate_angles(job_id, data_url, old_centroids_path, base_vector_x, base_vector_y):
    """
    Performs the pre processing of a general rotational clustering algorithm,
    i.e. the angle calculations.

    We take the data and centroids and calculate the centroid and data angles.
    """

    data_file_path = './static/angle-calculation/rotational-clustering/data' \
                     + str(job_id) + '.txt'
    centroids_file_path = './static/angle-calculation/rotational-clustering/centroids' \
                          + str(job_id) + '.txt'
    centroid_angles_file_path = './static/angle-calculation/rotational-clustering/centroid_angles' \
                                + str(job_id) + '.txt'
    data_angles_file_path = './static/angle-calculation/rotational-clustering/data_angles' \
                            + str(job_id) + '.txt'

    base_vector = np.array([base_vector_x, base_vector_y])

    # create working folder if not exist
    FileService.create_folder_if_not_exist('./static/angle-calculation/rotational-clustering/')

    # delete old files if exist
    FileService.delete_if_exist(data_file_path,
                                centroids_file_path,
                                centroid_angles_file_path,
                                data_angles_file_path)

    # download the data and store it locally
    await FileService.download_to_file(data_url, data_file_path)
    copyfile(old_centroids_path, centroids_file_path)

    # deserialize the data
    data = np.loadtxt(data_file_path)
    centroids = np.loadtxt(centroids_file_path)

    # map data and centroids to standardized unit sphere
    data = DataProcessingService.normalize(DataProcessingService.standardize(data))
    centroids = DataProcessingService.normalize(DataProcessingService.standardize(centroids))

    # calculate the angles
    data_angles = DataProcessingService.calculate_angles(data, base_vector)
    centroid_angles = DataProcessingService.calculate_angles(centroids, base_vector)

    # serialize the data
    np.savetxt(data_angles_file_path, data_angles)
    np.savetxt(centroid_angles_file_path, centroid_angles)

    return data_angles_file_path, centroid_angles_file_path


async def generate_negative_rotation_circuits(job_id, data_angles_path, centroid_angles_path, max_qubits=5):
    """
    Generates the negative rotation clustering quantum circuits.

    We take the data and centroid angles and return a url to a file with the
    quantum circuits as qasm strings.
    """

    data_angles_file_path = './static/circuit-generation/negative-rotation-clustering/data_angles' \
                            + str(job_id) + '.txt'
    centroid_angles_file_path = './static/circuit-generation/negative-rotation-clustering/centroid_angles' \
                                + str(job_id) + '.txt'
    circuits_file_path = './static/circuit-generation/negative-rotation-clustering/circuits' \
                         + str(job_id) + '.txt'

    # create working folder if not exist
    FileService.create_folder_if_not_exist('./static/circuit-generation/negative-rotation-clustering/')

    # delete old files if exist
    FileService.delete_if_exist(data_angles_file_path, centroid_angles_file_path, circuits_file_path)

    # copy data
    copyfile(data_angles_path, data_angles_file_path)
    copyfile(centroid_angles_path, centroid_angles_file_path)

    # deserialize the data and centroid angles
    data_angles = np.loadtxt(data_angles_file_path)
    centroid_angles = np.loadtxt(centroid_angles_file_path)

    # perform circuit generation
    circuits = ClusteringCircuitGenerator.generate_negative_rotation_clustering(max_qubits,
                                                                                data_angles,
                                                                                centroid_angles)

    # serialize the quantum circuits
    QiskitSerializer.serialize(circuits, circuits_file_path)

    return circuits_file_path


async def execute_negative_rotation_circuits(job_id, circuits_path, k, backend_name, token, shots=8192):
    """
    Executes the negative rotation clustering algorithm given the generated
    quantum circuits.
    """

    circuits_file_path = './static/circuit-execution/negative-rotation-clustering/circuits' \
                         + str(job_id) + '.txt'
    cluster_mapping_file_path = './static/circuit-execution/negative-rotation-clustering/cluster_mapping' \
                                + str(job_id) + '.txt'

    # create working folder if not exist
    FileService.create_folder_if_not_exist('./static/circuit-execution/negative-rotation-clustering/')

    # delete old files if exist
    FileService.delete_if_exist(circuits_file_path)

    # copy circuits
    copyfile(circuits_path, circuits_file_path)

    # deserialize the circuits
    circuits = QiskitSerializer.deserialize(circuits_file_path)

    # create the quantum backend
    backend = QuantumBackendFactory.create_backend(backend_name, token)

    # execute the circuits
    cluster_mapping = NegativeRotationClusteringService.execute_negative_rotation_clustering(circuits,
                                                                                             k,
                                                                                             backend,
                                                                                             shots)

    # serialize the data
    np.savetxt(cluster_mapping_file_path, cluster_mapping)

    return cluster_mapping_file_path


async def calculate_centroids(job_id, data_url, cluster_mapping_path, old_centroids_path):
    """
    Performs the post processing of a general rotational clustering algorithm,
    i.e. the centroid calculations.

    We take the cluster mapping, data and old centroids and calculate the
    new centroids.
    """

    data_file_path = './static/centroid-calculation/rotational-clustering/data' \
                     + str(job_id) + '.txt'
    cluster_mapping_file_path = './static/centroid-calculation/rotational-clustering/cluster_mapping' \
                                + str(job_id) + '.txt'
    old_centroids_file_path = './static/centroid-calculation/rotational-clustering/old_centroids' \
                              + str(job_id) + '.txt'
    centroids_file_path = './static/centroid-calculation/rotational-clustering/centroids' \
                          + str(job_id) + '.txt'

    # create working folder if not exist
    FileService.create_folder_if_not_exist('./static/centroid-calculation/rotational-clustering/')

    # delete old files if exist
    FileService.delete_if_exist(data_file_path, cluster_mapping_file_path, old_centroids_file_path)

    # download and copy the data
    await FileService.download_to_file(data_url, data_file_path)
    copyfile(cluster_mapping_path, cluster_mapping_file_path)
    copyfile(old_centroids_path, old_centroids_file_path)

    # deserialize the data
    data = np.loadtxt(data_file_path)
    cluster_mapping = np.loadtxt(cluster_mapping_file_path)
    old_centroids = np.loadtxt(old_centroids_file_path)

    # map data and centroids to standardized unit sphere
    data = DataProcessingService.normalize(DataProcessingService.standardize(data))
    old_centroids = DataProcessingService.normalize(DataProcessingService.standardize(old_centroids))

    # calculate new centroids
    centroids = DataProcessingService.calculate_centroids(cluster_mapping, old_centroids, data)

    # serialize the data
    np.savetxt(centroids_file_path, centroids)

    return centroids_file_path


async def check_convergence(job_id, new_centroids_path, old_centroids_path, eps=0.0001):
    """
    Performs the convergence check for a general KMeans clustering algorithm.

    We take the old and new centroids, calculate their pairwise distance and sum them up
    and divide it by k.

    If the resulting value is less then the given eps, we return convergence, if not,
    we return not converged.
    """

    old_centroids_file_path = './static/convergence-check/old_centroids' + str(job_id) + '.txt'
    new_centroids_file_path = './static/convergence-check/new_centroids' + str(job_id) + '.txt'

    # create working folder if not exist
    FileService.create_folder_if_not_exist('./static/convergence-check/')

    # delete old files if exist
    FileService.delete_if_exist(old_centroids_file_path, new_centroids_file_path)

    # download the data and store it locally
    copyfile(old_centroids_path, old_centroids_file_path)
    copyfile(new_centroids_path, new_centroids_file_path)

    # deserialize the data
    old_centroids = np.loadtxt(old_centroids_file_path)
    new_centroids = np.loadtxt(new_centroids_file_path)

    # check convergence
    distance = ConvergenceCalculationService.calculate_averaged_euclidean_distance(old_centroids, new_centroids)
    convergence = distance < eps

    return convergence, distance


def get_colors(k):
    """
    Return k colors in a list. We choose from 7 different colors.
    If k > 7 we choose colors more than once.
    """

    base_colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    colors = []
    index = 1
    for i in range(0, k):
        if index % (len(base_colors) + 1) == 0:
            index = 1
        colors.append(base_colors[index - 1])
        index += 1
    return colors


def plot_data(data_lists, data_names, title, circle=False):
    """
    Plot all the data points with optional an unit circle.
    We expect to have data lists as a list of cluster points, i.e.
    data_lists =
    [ [ [x_0, y_0], [x_1, y_1], ... ], [x_0, y_0], [x_1, y_1], ... ] ]
    [      points of cluster 1       ,    points of cluster 2        ]
    data_names = [ "Cluster1", "Cluster2" ].
    """

    plt.clf()
    plt.figure(figsize=(7, 7), dpi=80)
    unit_circle_plot = plt.Circle((0, 0), 1.0, color='k', fill=False)
    ax = plt.gca()
    ax.cla()
    ax.set_xlim(-1.5, +1.5)
    ax.set_ylim(-1.5, +1.5)
    ax.set_title(title)
    if circle:
        ax.add_artist(unit_circle_plot)
    colors = get_colors(len(data_lists))
    for i in range(0, len(data_lists)):
        ax.scatter([data_points[0] for data_points in data_lists[i]],
                   [dataPoints[1] for dataPoints in data_lists[i]],
                   color=colors[i],
                   label=data_names[i])

    FileService.create_folder_if_not_exist('./static/plots/')
    plt.savefig('./static/plots/' + title, dpi=300, bbox_inches='tight')
    return


def plot(data_raw, data, cluster_mapping, k):
    """
    Prepare data in order to plot.
    """

    data_texts = []
    clusters = dict()
    clusters_raw = dict()

    for i in range(0, cluster_mapping.shape[0]):
        cluster_number = int(cluster_mapping[i])
        if cluster_number not in clusters:
            clusters[cluster_number] = []
            clusters_raw[cluster_number] = []

        clusters[cluster_number].append(data[i])
        clusters_raw[cluster_number].append(data_raw[i])

    # add missing clusters that have no elements
    for i in range(0, k):
        if i not in clusters:
            clusters[i] = []

    clusters_plot = []
    clusters_raw_plot = []

    for i in range(0, k):
        clusters_plot.append([])
        clusters_raw_plot.append([])
        for j in range(0, len(clusters[i])):
            clusters_plot[i].append(clusters[i][j])
            clusters_raw_plot[i].append(clusters_raw[i][j])

    for i in range(0, k):
        data_texts.append("Cluster" + str(i))

    plot_data(clusters_plot, data_texts, "Preprocessed Data", circle=True)
    plot_data(clusters_raw_plot, data_texts, "Raw Data")


async def plot_data_from_urls(data_url, cluster_mapping_url, k):
    # create paths
    plot_folder_path = './static/test/plot/'
    data_file_path = plot_folder_path + 'data.txt'
    cluster_mapping_file_path = plot_folder_path + 'cluster_mapping.txt'

    # create folder and delete old files
    FileService.create_folder_if_not_exist(plot_folder_path)
    FileService.delete_if_exist(data_file_path, cluster_mapping_file_path)

    # download data
    await FileService.download_to_file(data_url, data_file_path)
    copyfile(cluster_mapping_url, cluster_mapping_file_path)

    # deserialize data
    data = np.loadtxt(data_file_path)
    cluster_mapping = np.loadtxt(cluster_mapping_file_path)

    # prepare plot data
    data_preprocessed = DataProcessingService.normalize(DataProcessingService.standardize(data))

    plot(data, data_preprocessed, cluster_mapping, k)


async def main():
    """
    This method tests the entire workflow within a python script.
    """

    k = 2
    job_id = 1
    eps = 0.001
    max_runs = 10
    data_url = 'https://raw.githubusercontent.com/UST-QuAntiL/QuantME-UseCases/master/2021-icws/data/embedding.txt'
    backend_name = 'aer_qasm_simulator'
    token = ""
    print('Starting test script...')

    old_centroids_path = await initialize_centroids(job_id, k)
    print('Path of old centroids: ' + str(old_centroids_path))

    for i in range(1, max_runs + 1):
        print('Running iteration: ' + str(i))

        data_angles_path, centroid_angles_path = await calculate_angles(job_id, data_url, old_centroids_path, -0.7071,
                                                                        0.7071)

        circuits_path = await generate_negative_rotation_circuits(job_id, data_angles_path, centroid_angles_path)
        cluster_mapping_path = await execute_negative_rotation_circuits(job_id, circuits_path, k, backend_name, token)
        new_centroids_path = await calculate_centroids(job_id, data_url, cluster_mapping_path, old_centroids_path)
        convergence, distance = await check_convergence(job_id, new_centroids_path, old_centroids_path, eps)

        # replace old centroids with new centroids
        old_centroids_path = new_centroids_path

        print('Iteration ' + str(i) + ' / ' + str(max_runs) + ' Distance = ' + str(distance))
        if convergence:
            print('Converged!')
            break

    # plot the results
    await plot_data_from_urls(data_url, cluster_mapping_path, k)


if __name__ == "__main__":
    asyncio.run(main())
