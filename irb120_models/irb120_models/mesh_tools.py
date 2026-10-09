"""
Triangle-mesh I/O and rigid-body properties, numpy only.

The object meshes come out of CAD as glTF (vtk reads them for the perception
contact checks). URDF consumers want STL: RViz loads either, but MuJoCo's URDF
import only takes STL/OBJ, and ``.obj`` is gitignored in this repo. So the
glTF stays the source and the STL is generated from it.

trimesh would do all of this, but it isn't installed on the lab PC and the
package would then need a pip dependency rosdep can't resolve. The volume
integrals below are the standard signed-tetrahedron ones; they assume a closed,
consistently wound surface, which CAD exports are. ``mass_properties`` reports
the signed volume so a broken mesh shows up as a negative or implausible value
instead of silently wrong inertia.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
import struct

import numpy as np

_COMPONENT_DTYPES = {
    5120: np.int8, 5121: np.uint8, 5122: np.int16,
    5123: np.uint16, 5125: np.uint32, 5126: np.float32,
}
_TYPE_SIZES = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def load_triangles(path: str | Path) -> np.ndarray:
    """Return an (N, 3, 3) array of triangles in metres from a .gltf or .stl file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.gltf':
        return _load_gltf(path)
    if suffix == '.stl':
        return _load_stl(path)
    raise ValueError(f'Unsupported mesh format {suffix!r} ({path}); use .gltf or .stl')


def stl_bytes(triangles: np.ndarray, header: str = '') -> bytes:
    """Binary STL for the triangles, with per-face normals from the winding."""
    tri = np.asarray(triangles, dtype=np.float64)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    record = np.zeros(len(tri), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('attr', '<u2')])
    record['n'] = normals
    record['v'] = tri
    return (header.encode('ascii', 'replace')[:80].ljust(80, b' ')
            + struct.pack('<I', len(tri)) + record.tobytes())


def write_stl(path: str | Path, triangles: np.ndarray, header: str = '') -> None:
    Path(path).write_bytes(stl_bytes(triangles, header))


def mass_properties(triangles: np.ndarray, mass: float | None = None) -> dict:
    """
    Volume, centroid and inertia of the solid the closed surface bounds.

    Uniform density is assumed. If ``mass`` is given the inertia is scaled to it;
    otherwise it is for unit density. The inertia is about the centroid, in the
    mesh frame.
    """
    tri = np.asarray(triangles, dtype=np.float64)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    vol_i = np.einsum('ij,ij->i', a, np.cross(b, c)) / 6.0
    volume = float(vol_i.sum())
    if abs(volume) < 1e-12:
        raise ValueError('Mesh encloses no volume; is it a closed surface?')
    centroid = (vol_i[:, None] * (a + b + c)).sum(axis=0) / (4.0 * volume)

    # Second moments of each signed tetrahedron (origin, a, b, c):
    # integral of x_i x_j = vol/20 * (sum_k v_ki v_kj + (sum_k v_ki)(sum_k v_kj)).
    s = a + b + c
    second = sum(np.einsum('n,ni,nj->ij', vol_i, v, v) for v in (a, b, c, s)) / 20.0
    second -= volume * np.outer(centroid, centroid)  # move to the centroid
    inertia = np.trace(second) * np.eye(3) - second
    density = 1.0 if mass is None else mass / volume
    return {
        'volume': volume,
        'centroid': centroid,
        'inertia': inertia * density,
        'bounds': np.stack([tri.reshape(-1, 3).min(axis=0), tri.reshape(-1, 3).max(axis=0)]),
    }


# --------------------------------------------------------------------------- glTF

def _load_gltf(path: Path) -> np.ndarray:
    doc = json.loads(path.read_text())
    buffers = [_read_buffer(path, b) for b in doc['buffers']]

    def accessor(index: int) -> np.ndarray:
        acc = doc['accessors'][index]
        view = doc['bufferViews'][acc['bufferView']]
        dtype = np.dtype(_COMPONENT_DTYPES[acc['componentType']]).newbyteorder('<')
        width = _TYPE_SIZES[acc['type']]
        stride = view.get('byteStride') or dtype.itemsize * width
        start = view.get('byteOffset', 0) + acc.get('byteOffset', 0)
        data = buffers[view['buffer']]
        if stride == dtype.itemsize * width:  # tightly packed: one read
            return np.frombuffer(data, dtype=dtype, count=acc['count'] * width,
                                 offset=start).reshape(acc['count'], width)
        return np.stack([np.frombuffer(data, dtype=dtype, count=width, offset=start + k * stride)
                         for k in range(acc['count'])])

    scene = doc['scenes'][doc.get('scene', 0)]
    triangles = []

    def visit(node_index: int, parent: np.ndarray) -> None:
        node = doc['nodes'][node_index]
        world = parent @ _node_matrix(node)
        if 'mesh' in node:
            for prim in doc['meshes'][node['mesh']]['primitives']:
                if prim.get('mode', 4) != 4:
                    raise ValueError(f'{path}: only triangle primitives are supported')
                verts = accessor(prim['attributes']['POSITION']).astype(np.float64)
                verts = verts @ world[:3, :3].T + world[:3, 3]
                if 'indices' in prim:
                    idx = accessor(prim['indices']).reshape(-1, 3).astype(np.int64)
                else:
                    idx = np.arange(len(verts)).reshape(-1, 3)
                triangles.append(verts[idx])
        for child in node.get('children', []):
            visit(child, world)

    for root in scene['nodes']:
        visit(root, np.eye(4))
    if not triangles:
        raise ValueError(f'{path}: no triangle meshes found')
    return np.concatenate(triangles)


def _read_buffer(path: Path, buffer: dict) -> bytes:
    uri = buffer['uri']
    if uri.startswith('data:'):
        return base64.b64decode(uri.split(',', 1)[1])
    return (path.parent / uri).read_bytes()


def _node_matrix(node: dict) -> np.ndarray:
    if 'matrix' in node:
        return np.asarray(node['matrix'], dtype=np.float64).reshape(4, 4).T  # column-major
    m = np.eye(4)
    x, y, z, w = node.get('rotation', (0.0, 0.0, 0.0, 1.0))
    m[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]) @ np.diag(node.get('scale', (1.0, 1.0, 1.0)))
    m[:3, 3] = node.get('translation', (0.0, 0.0, 0.0))
    return m


# --------------------------------------------------------------------------- STL

def _load_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) >= 84:
        count = struct.unpack_from('<I', data, 80)[0]
        if 84 + 50 * count == len(data):
            record = np.frombuffer(data, offset=84, count=count,
                                   dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('attr', '<u2')])
            return record['v'].astype(np.float64)
    verts = [[float(v) for v in line.split()[1:4]]
             for line in data.decode('ascii', 'replace').splitlines()
             if line.strip().startswith('vertex')]
    if not verts or len(verts) % 3:
        raise ValueError(f'{path}: not a valid binary or ASCII STL')
    return np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)
