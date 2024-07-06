"""
I/O for Abaqus inp files.
"""

import pathlib
from itertools import count

import numpy as np
import logging
import os

from ..__about__ import __version__
from .._common import num_nodes_per_cell
from .._exceptions import ReadError
from .._files import open_file
from .._helpers import register_format
from .._mesh import CellBlock, Mesh

abaqus_to_meshio_type = {
    # trusses
    "T2D2": "line",
    "T2D2H": "line",
    "T2D3": "line3",
    "T2D3H": "line3",
    "T3D2": "line",
    "T3D2H": "line",
    "T3D3": "line3",
    "T3D3H": "line3",
    # beams
    "B21": "line",
    "B21H": "line",
    "B22": "line3",
    "B22H": "line3",
    "B31": "line",
    "B31H": "line",
    "B32": "line3",
    "B32H": "line3",
    "B33": "line3",
    "B33H": "line3",
    # surfaces
    "CPS4": "quad",
    "CPS4R": "quad",
    "S4": "quad",
    "S4R": "quad",
    "S4RS": "quad",
    "S4RSW": "quad",
    "S4R5": "quad",
    "S8R": "quad8",
    "S8R5": "quad8",
    "S9R5": "quad9",
    # "QUAD": "quad",
    # "QUAD4": "quad",
    # "QUAD5": "quad5",
    # "QUAD8": "quad8",
    # "QUAD9": "quad9",
    #
    "CPS3": "triangle",
    "STRI3": "triangle",
    "S3": "triangle",
    "S3R": "triangle",
    "S3RS": "triangle",
    "R3D3": "triangle",
    # "TRI7": "triangle7",
    # 'TRISHELL': 'triangle',
    # 'TRISHELL3': 'triangle',
    # 'TRISHELL7': 'triangle',
    #
    "STRI65": "triangle6",
    # 'TRISHELL6': 'triangle6',
    # volumes
    "C3D8": "hexahedron",
    "C3D8H": "hexahedron",
    "C3D8I": "hexahedron",
    "C3D8IH": "hexahedron",
    "C3D8R": "hexahedron",
    "C3D8RH": "hexahedron",
    # "HEX9": "hexahedron9",
    "C3D20": "hexahedron20",
    "C3D20H": "hexahedron20",
    "C3D20R": "hexahedron20",
    "C3D20RH": "hexahedron20",
    # "HEX27": "hexahedron27",
    #
    "C3D4": "tetra",
    "C3D4H": "tetra4",
    # "TETRA8": "tetra8",
    "C3D10": "tetra10",
    "C3D10H": "tetra10",
    "C3D10I": "tetra10",
    "C3D10M": "tetra10",
    "C3D10MH": "tetra10",
    # "TETRA14": "tetra14",
    #
    # "PYRAMID": "pyramid",
    "C3D6": "wedge",
    "C3D15": "wedge15",
    #
    # 4-node bilinear displacement and pore pressure
    "CAX4P": "quad",
    # 6-node quadratic
    "CPE6": "triangle6",
}
meshio_to_abaqus_type = {v: k for k, v in abaqus_to_meshio_type.items()}


def read(filename):
    """Reads a Abaqus inp file."""
    with open_file(filename, "r") as f:
        out = read_buffer(f)
    return out

def read_buffer(f):
    parts = read_buffer_multi(f)
    if isinstance(parts, Mesh):
        return parts
    else:
        # just return the first part
        partname = os.environ.get("MESHIO_ABAQUS_PART_NAME")
        if len(parts) > 1 and partname is None:
            keys = list(parts.keys())
            partname = keys[0]
            logging.warning(f"Only processing part/assembly '{partname}'. Use environment variable `MESHIO_ABAQUS_PART_NAME` to override. Available: {keys}")
        elif partname is None:
            partname = list(parts.keys())[0]
        if partname not in parts:
            raise RuntimeError(f"Invalid part/assembly name '{partname}'")
        return parts[partname]

def read_buffer_multi(f):
    # Initialize the optional data fields
    points = []
    cells = []
    cell_ids = []
    point_sets = {}
    cell_sets = {}
    cell_sets_element = {}  # Handle cell sets defined in ELEMENT
    cell_sets_element_order = []  # Order of keys is not preserved in Python 3.5
    field_data = {}
    cell_data = {}
    point_data = {}
    point_ids = None

    parts = {}
    part_name = None

    assemblies = {}

    line = f.readline()
    while True:
        if not line:  # EOF
            break

        # Comments
        if line.startswith("**"):
            line = f.readline()
            continue

        keyword = line.partition(",")[0].strip().replace("*", "").upper()
        if keyword == "PART":
            params_map = get_param_map(line, required_keys=["NAME"])
            print("reading  part", params_map["NAME"])
            part_name = params_map["NAME"]
            line = f.readline()
        elif keyword == "END PART":
            line = f.readline()
            m = Mesh(
                    points,
                    cells,
                    point_data=point_data,
                    cell_data=cell_data,
                    field_data=field_data,
                    # point_sets=point_sets,
                    # cell_sets=cell_sets,
                )
            parts[part_name] = m

            # reset all datastructures
            part_name = None
            points = []
            cells = []
            cell_ids = []
            point_sets = {}
            cell_sets = {}
            cell_sets_element = {}  # Handle cell sets defined in ELEMENT
            cell_sets_element_order = []  # Order of keys is not preserved in Python 3.5
            field_data = {}
            cell_data = {}
            point_data = {}
            point_ids = None
            # print("end part")
        elif keyword == "NODE":
            # print("read node")
            points, point_ids, line = _read_nodes(f)
        elif keyword == "ASSEMBLY":
            assemblies = _read_assembly(f, parts)
            line = f.readline()
        elif keyword == "ELEMENT":
            if point_ids is None:
                raise ReadError("Expected NODE before ELEMENT")
            params_map = get_param_map(line, required_keys=["TYPE"])
            cell_type, cells_data, ids, sets, line = _read_cells(
                f, params_map, point_ids
            )
            cells.append(CellBlock(cell_type, cells_data))
            cell_ids.append(ids)
            if sets:
                cell_sets_element.update(sets)
                cell_sets_element_order += list(sets.keys())
        elif keyword == "NSET":
            params_map = get_param_map(line, required_keys=["NSET"])
            set_ids, _, line = _read_set(f, params_map)
            name = params_map["NSET"]
            point_sets[name] = np.array(
                [point_ids[point_id] for point_id in set_ids], dtype="int32"
            )
        elif keyword == "ELSET":
            params_map = get_param_map(line, required_keys=["ELSET"])
            set_ids, set_names, line = _read_set(f, params_map)
            name = params_map["ELSET"]
            cell_sets[name] = []
            if set_ids.size:
                for cell_ids_ in cell_ids:
                    cell_sets_ = np.array(
                        [
                            cell_ids_[set_id]
                            for set_id in set_ids
                            if set_id in cell_ids_
                        ],
                        dtype="int32",
                    )
                    cell_sets[name].append(cell_sets_)
            elif set_names:
                for set_name in set_names:
                    if set_name in cell_sets.keys():
                        cell_sets[name].append(cell_sets[set_name])
                    elif set_name in cell_sets_element.keys():
                        cell_sets[name].append(cell_sets_element[set_name])
                    else:
                        raise ReadError(f"Unknown cell set '{set_name}'")
        elif keyword == "INCLUDE":
            # Splitting line to get external input file path (example: *INCLUDE,INPUT=wInclude_bulk.inp)
            ext_input_file = pathlib.Path(line.split("=")[-1].strip())
            if ext_input_file.exists() is False:
                cd = pathlib.Path(f.name).parent
                ext_input_file = cd / ext_input_file

            # Read contents from external input file into mesh object
            out = read(ext_input_file)

            # Merge contents of external file only if it is containing mesh data
            if len(out.points) > 0:
                points, cells = merge(
                    out,
                    points,
                    cells,
                    point_data,
                    cell_data,
                    field_data,
                    point_sets,
                    cell_sets,
                )

            line = f.readline()
        else:
            # There are just too many Abaqus keywords to explicitly skip them.
            line = f.readline()

    # Parse cell sets defined in ELEMENT
    for i, name in enumerate(cell_sets_element_order):
        # Not sure whether this case would ever happen
        if name in cell_sets.keys():
            cell_sets[name][i] = cell_sets_element[name]
        else:
            cell_sets[name] = []
            for ic in range(len(cells)):
                cell_sets[name].append(
                    cell_sets_element[name] if i == ic else np.array([], dtype="int32")
                )

    if assemblies:
        return assemblies

    if parts:
        return parts

    return Mesh(
        points,
        cells,
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
        point_sets=point_sets,
        cell_sets=cell_sets,
    )


def _read_nodes(f):
    points = []
    point_ids = {}
    counter = 0
    while True:
        line = f.readline()
        if not line or line.startswith("*"):
            break
        if line.strip() == "":
            continue

        line = line.strip().split(",")
        point_id, coords = line[0], line[1:]
        point_ids[int(point_id)] = counter
        points.append([float(x) for x in coords])
        counter += 1

    return np.array(points, dtype=float), point_ids, line


def _read_cells(f, params_map, point_ids):
    etype = params_map["TYPE"]
    if etype not in abaqus_to_meshio_type.keys():
        raise ReadError(f"Element type not available: {etype}")

    cell_type = abaqus_to_meshio_type[etype]
    # ElementID + NodesIDs
    num_data = num_nodes_per_cell[cell_type] + 1

    idx = []
    while True:
        line = f.readline()
        if not line or line.startswith("*"):
            break
        line = line.strip()
        if line == "":
            continue
        idx += [int(k) for k in filter(None, line.split(","))]

    # Check for expected number of data
    if len(idx) % num_data != 0:
        raise ReadError("Expected number of data items does not match element type")

    idx = np.array(idx).reshape((-1, num_data))
    cell_ids = dict(zip(idx[:, 0], count(0)))
    cells = np.array([[point_ids[node] for node in elem] for elem in idx[:, 1:]])

    cell_sets = (
        {params_map["ELSET"]: np.arange(len(cells), dtype="int32")}
        if "ELSET" in params_map.keys()
        else {}
    )

    return cell_type, cells, cell_ids, cell_sets, line


def merge(
    mesh, points, cells, point_data, cell_data, field_data, point_sets, cell_sets
):
    """
    Merge Mesh object into existing containers for points, cells, sets, etc..

    :param mesh:
    :param points:
    :param cells:
    :param point_data:
    :param cell_data:
    :param field_data:
    :param point_sets:
    :param cell_sets:
    :type mesh: Mesh
    """
    ext_points = np.array([p for p in mesh.points])

    if len(points) > 0:
        new_point_id = points.shape[0]
        # new_cell_id = len(cells) + 1
        points = np.concatenate([points, ext_points])
    else:
        # new_cell_id = 0
        new_point_id = 0
        points = ext_points

    cnt = 0
    for c in mesh.cells:
        new_data = np.array([d + new_point_id for d in c.data])
        cells.append(CellBlock(c.type, new_data))
        cnt += 1

    # The following aren't currently included in the abaqus parser, and are therefore
    # excluded?
    # point_data.update(mesh.point_data)
    # cell_data.update(mesh.cell_data)
    # field_data.update(mesh.field_data)

    # Update point and cell sets to account for change in cell and point ids
    for key, val in mesh.point_sets.items():
        point_sets[key] = [x + new_point_id for x in val]

    # Todo: Add support for merging cell sets
    # cellblockref = [[] for i in range(cnt-new_cell_id)]
    # for key, val in mesh.cell_sets.items():
    #     cell_sets[key] = cellblockref + [np.array([x for x in val[0]])]

    return points, cells


def get_param_map(word, required_keys=None):
    """
    get the optional arguments on a line

    Example
    -------
    >>> word = 'elset,instance=dummy2,generate'
    >>> params = get_param_map(word, required_keys=['instance'])
    params = {
        'elset' : None,
        'instance' : 'dummy2,
        'generate' : None,
    }
    """
    if required_keys is None:
        required_keys = []
    words = word.split(",")
    param_map = {}
    for wordi in words:
        if "=" not in wordi:
            key = wordi.strip().upper()
            value = None
        else:
            sword = wordi.split("=")
            if len(sword) != 2:
                raise ReadError(sword)
            key = sword[0].strip().upper()
            value = sword[1].strip()
        param_map[key] = value

    msg = ""
    for key in required_keys:
        if key not in param_map:
            msg += f"{key} not found in {word}\n"
    if msg:
        raise RuntimeError(msg)
    return param_map


def _read_set(f, params_map):
    set_ids = []
    set_names = []
    while True:
        line = f.readline()
        if not line or line.startswith("*"):
            break
        if line.strip() == "":
            continue

        line = line.strip().strip(",").split(",")
        if line[0].isnumeric():
            set_ids += [int(k) for k in line]
        else:
            set_names.append(line[0])

    set_ids = np.array(set_ids, dtype="int32")
    if "GENERATE" in params_map:
        if len(set_ids) != 3:
            raise ReadError(set_ids)
        set_ids = np.arange(set_ids[0], set_ids[1] + 1, set_ids[2], dtype="int32")
    return set_ids, set_names, line

def _read_assembly(f, parts):
    assemblies = {}

    line = f.readline()
    while True:
        if not line:  # EOF
            break

        # Comments
        if line.startswith("**"):
            line = f.readline()
            continue

        keyword = line.partition(",")[0].strip().replace("*", "").upper()
        if keyword == "END ASSEMBLY":
            break
        elif keyword == "INSTANCE":
            params_map = get_param_map(line, required_keys=["NAME", "PART"])
            if params_map["PART"] not in parts:
                raise RuntimeError("Unknown part identified in assembly", params_map["PART"])
            instance = parts[params_map["PART"]]
            translation, rotation, line = _read_instance(f)
            if rotation and len(rotation) == 7:
                instance.points = _transform_coordinates(instance.points, np.asarray(rotation[:3]), np.asarray(rotation[3:6]), rotation[6]) + np.asarray(translation)
            assemblies[params_map["NAME"]] = instance
        else:
            line = f.readline()
 
    return assemblies

def _read_instance(f):
    values = []
    while True:
        line = f.readline()
        if not line or line.startswith("*"):
            break
        if line.strip() == "":
            continue
        line = line.strip().strip(",").split(",")
        values += [float(k) for k in line]
    return values[:3], values[3:], line

def _transform_coordinates(coords, P1, P2, theta_deg):
    """
    Transform the given coordinates by translating, rotating around a specified axis,
    and then translating back.

    Parameters:
    coords (np.ndarray): Array of coordinates to be transformed (Nx3).
    P1 (np.ndarray): Reference point (origin of the transformation) as a 1x3 array.
    P2 (np.ndarray): Direction point (defines the axis of rotation) as a 1x3 array.
    theta_deg (float): Rotation angle in degrees.

    Returns:
    np.ndarray: Transformed coordinates.
    """
    
    # Compute the unit vector for the axis of rotation
    u = P2 - P1
    u = u / np.linalg.norm(u)

    # Convert the angle to radians
    theta_rad = np.deg2rad(theta_deg)

    # Define the rotation matrix components
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    ux, uy, uz = u

    # Compute the rotation matrix
    R = np.array([
        [cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
        [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
        [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
    ])

    # Define the translation matrix (to the origin P1)
    T = np.eye(4)
    T[:3, 3] = P1

    # Define the inverse translation matrix (back to the original position)
    T_inv = np.eye(4)
    T_inv[:3, 3] = -P1

    # Combine the rotation matrix into a 4x4 matrix for homogeneous coordinates
    R_homogeneous = np.eye(4)
    R_homogeneous[:3, :3] = R

    # Compute the combined transformation matrix
    M = np.dot(np.dot(T, R_homogeneous), T_inv)

    # Convert input coordinates to homogeneous coordinates
    coords_homogeneous = np.ones((coords.shape[0], 4))
    coords_homogeneous[:, :3] = coords

    # Apply the transformation
    transformed_coords_homogeneous = np.dot(M, coords_homogeneous.T).T

    # Convert back to 3D coordinates
    transformed_coords = transformed_coords_homogeneous[:, :3]

    return transformed_coords

def write(
    filename, mesh: Mesh, float_fmt: str = ".16e", translate_cell_names: bool = True
) -> None:
    with open_file(filename, "wt") as f:
        f.write("*HEADING\n")
        f.write("Abaqus DataFile Version 6.14\n")
        f.write(f"written by meshio v{__version__}\n")
        f.write("*NODE\n")
        fmt = ", ".join(["{}"] + ["{:" + float_fmt + "}"] * mesh.points.shape[1]) + "\n"
        for k, x in enumerate(mesh.points):
            f.write(fmt.format(k + 1, *x))
        eid = 0
        for cell_block in mesh.cells:
            cell_type = cell_block.type
            node_idcs = cell_block.data
            name = (
                meshio_to_abaqus_type[cell_type] if translate_cell_names else cell_type
            )
            f.write(f"*ELEMENT, TYPE={name}\n")
            for row in node_idcs:
                eid += 1
                nids_strs = (str(nid + 1) for nid in row.tolist())
                f.write(str(eid) + "," + ",".join(nids_strs) + "\n")

        nnl = 8
        offset = 0
        for ic in range(len(mesh.cells)):
            for k, v in mesh.cell_sets.items():
                if len(v[ic]) > 0:
                    els = [str(i + 1 + offset) for i in v[ic]]
                    f.write(f"*ELSET, ELSET={k}\n")
                    f.write(
                        ",\n".join(
                            ",".join(els[i : i + nnl]) for i in range(0, len(els), nnl)
                        )
                        + "\n"
                    )
            offset += len(mesh.cells[ic].data)

        for k, v in mesh.point_sets.items():
            nds = [str(i + 1) for i in v]
            f.write(f"*NSET, NSET={k}\n")
            f.write(
                ",\n".join(",".join(nds[i : i + nnl]) for i in range(0, len(nds), nnl))
                + "\n"
            )

        # https://github.com/nschloe/meshio/issues/747#issuecomment-643479921
        # f.write("*END")


register_format("abaqus", [".inp"], read, {"abaqus": write})
