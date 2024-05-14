"""Helper script to test some lane graph utils."""
# pylint: skip-file

import copy
import itertools
import sys

import abstract_graph
import abstract_graph as ab
import matplotlib.pyplot as plt
import networkx as nx


def main(pickle_path):
    print("init")

    # Read the graph from the provided gpickle file
    print("gpickle read")
    read_lanegraph = nx.read_gpickle(pickle_path)
    read_lanegraph1 = copy.deepcopy(read_lanegraph)

    print("Remove Redundancy 1")
    pred_lanegraph1 = abstract_graph.remove_redundancy(read_lanegraph)

    print("Remove Redundancy 2")
    pred_lanegraph = abstract_graph.remove_single_ends(pred_lanegraph1)

    # for eps in [float(x) for x in range(4, 10)]:

    #     for min_samples in range(1, 4):

    eps = 18.0
    min_samples = 4

    print()
    print(" ------------------------ ")
    print(f"Combination: EPS: {eps} MIN_SAMPLES: {min_samples}")

    # # Call the desired function on the graph
    intersections, streets, out_graph = ab.generate_abstract_graph(pred_lanegraph, eps, min_samples)

    # Print the results (or you can handle them in whatever way you need)
    print(f"Intersections:  {len(intersections)} ")
    print(f"Streets:        {len(streets)}  ")

    # if 6 < len(intersections) < 10 and out_graph.nodes[200181].get('spatial') == '1':

    #     print("Good one. ")

    # out_graph = pred_lanegraph

    # Get positions
    pos = nx.get_node_attributes(out_graph, 'pos')

    # # Get spatial attribute and color nodes based on it
    node_colors = ['red' if out_graph.nodes[node].get('spatial') == '1' else 'blue' for node in out_graph.nodes]

    # Draw nodes
    nx.draw_networkx_nodes(out_graph, pos, node_color=node_colors, node_size=20)
    # nx.draw_networkx_labels(out_graph, pos)
    nx.draw_networkx_edges(out_graph, pos)

    # nx.write_gpickle(out_graph,
    #                     f"annotated_lanegraph_eps-{eps}_min-samples-{min_samples}.gpickle")

    # Show plot
    plt.show()
    # plt.savefig(f"annotated_spatial_regions_eps-{eps}_min-samples-{min_samples}.png")  #

    print(" ------------------------ ")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: script_name.py <path_to_gpickle_file>")
        sys.exit(1)

    print("start")

    pickle_path = sys.argv[1]
    print("start2")
    main(pickle_path)
