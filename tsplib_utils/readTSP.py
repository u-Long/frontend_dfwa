import numpy as np
import math

def read_tsp_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Initialize variables
    name = None
    problem_type = None
    comment = None
    dimension = None
    edge_weight_type = None
    edge_weight_format = None
    display_data_type = None
    node_coord_section = False
    edge_weight_section = False
    nodes = {}
    edge_weights = []
    distance_matrix = None
    
    # Helper functions for distance calculations
    def euclidean_distance(coord1, coord2):
        return round(math.hypot(coord1[0] - coord2[0], coord1[1] - coord2[1]))
    
    def geo_distance(coord1, coord2):
        # Convert degrees to radians
        def deg2rad(deg):
            return math.pi * (deg + 0.0) / 180.0

        lat1, lon1 = coord1
        lat2, lon2 = coord2

        lat1_rad = deg2rad(lat1)
        lon1_rad = deg2rad(lon1)
        lat2_rad = deg2rad(lat2)
        lon2_rad = deg2rad(lon2)

        RRR = 6378.388  # Radius of the Earth in km

        q1 = math.cos(lon1_rad - lon2_rad)
        q2 = math.cos(lat1_rad - lat2_rad)
        q3 = math.cos(lat1_rad + lat2_rad)
        return int(RRR * math.acos(0.5 * ((1 + q2) * q1 - (1 - q2) * q3)) + 1)

    # Read the file line by line
    for line in lines:
        line = line.strip()
        if not line or line.startswith('EOF'):
            break
        if line.startswith('NAME'):
            name = line.split(':')[1].strip()
        elif line.startswith('TYPE'):
            problem_type = line.split(':')[1].strip()
        elif line.startswith('COMMENT'):
            comment = line.split(':')[1].strip()
        elif line.startswith('DIMENSION'):
            dimension = int(line.split(':')[1].strip())
            distance_matrix = np.zeros((dimension, dimension))
        elif line.startswith('EDGE_WEIGHT_TYPE'):
            edge_weight_type = line.split(':')[1].strip()
        elif line.startswith('EDGE_WEIGHT_FORMAT'):
            edge_weight_format = line.split(':')[1].strip()
        elif line.startswith('DISPLAY_DATA_TYPE'):
            display_data_type = line.split(':')[1].strip()
        elif line.startswith('NODE_COORD_SECTION'):
            node_coord_section = True
            continue
        elif line.startswith('EDGE_WEIGHT_SECTION'):
            edge_weight_section = True
            continue
        elif node_coord_section:
            parts = line.split()
            if len(parts) >= 3:
                idx = int(parts[0]) - 1  # Assuming nodes are 1-indexed
                x = float(parts[1])
                y = float(parts[2])
                nodes[idx] = (x, y)
            if len(nodes) == dimension:
                node_coord_section = False
        elif edge_weight_section:
            edge_weights.extend(map(float, line.split()))
            if len(edge_weights) >= dimension * dimension:
                edge_weight_section = False
    
    # Build the distance matrix
    if edge_weight_type == 'EXPLICIT':
        if edge_weight_format == 'FULL_MATRIX':
            if len(edge_weights) != dimension * dimension:
                raise ValueError("Edge weights do not match the dimension.")
            distance_matrix = np.array(edge_weights).reshape(dimension, dimension)
        elif edge_weight_format == 'LOWER_DIAG_ROW':
            idx = 0
            for i in range(dimension):
                for j in range(i + 1):
                    distance_matrix[i][j] = edge_weights[idx]
                    distance_matrix[j][i] = edge_weights[idx]
                    idx += 1
        elif edge_weight_format == 'UPPER_ROW':
            idx = 0
            for i in range(dimension):
                for j in range(i + 1, dimension):
                    distance_matrix[i][j] = edge_weights[idx]
                    distance_matrix[j][i] = edge_weights[idx]
                    idx += 1
        else:
            raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {edge_weight_format}")
    elif edge_weight_type in ['EUC_2D', 'CEIL_2D']:
        for i in range(dimension):
            for j in range(dimension):
                if i != j:
                    dist = euclidean_distance(nodes[i], nodes[j])
                    if edge_weight_type == 'CEIL_2D':
                        dist = math.ceil(dist)
                    distance_matrix[i][j] = dist
    elif edge_weight_type == 'GEO':
        for i in range(dimension):
            for j in range(dimension):
                if i != j:
                    dist = geo_distance(nodes[i], nodes[j])
                    distance_matrix[i][j] = dist
    else:
        raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type}")
    
    return {
        "name": name,
        "type": problem_type,
        "comment": comment,
        "dimension": dimension,
        "edge_weight_type": edge_weight_type,
        "edge_weight_format": edge_weight_format,
        "display_data_type": display_data_type,
        "nodes": nodes,
        "distance_matrix": distance_matrix
    }

if __name__ == "__main__":
    # Example usage
    filename = 'data/rat575.tsp'
    data = read_tsp_file(filename)
    print(data)

