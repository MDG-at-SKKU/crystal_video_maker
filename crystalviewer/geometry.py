"""
Geometry utilities for Crystal Viewer
Geometric calculations, transformations, and 3D vector operations
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import math
from functools import lru_cache

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .config import ATOMIC_RADII, get_atomic_radius
from .utils import vectorized_distance, safe_divide


class Vector3D:
    """3D Vector class with optimized operations"""

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x, self.y, self.z = x, y, z

    def __add__(self, other: "Vector3D") -> "Vector3D":
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3D") -> "Vector3D":
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3D":
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> "Vector3D":
        return self * safe_divide(1.0, scalar)

    def dot(self, other: "Vector3D") -> float:
        """Dot product"""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3D") -> "Vector3D":
        """Cross product"""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def magnitude(self) -> float:
        """Vector magnitude"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> "Vector3D":
        """Return normalized vector"""
        mag = self.magnitude()
        return self / mag if mag > 0 else Vector3D(0, 0, 0)

    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple"""
        return (self.x, self.y, self.z)

    def to_list(self) -> List[float]:
        """Convert to list"""
        return [self.x, self.y, self.z]

    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> "Vector3D":
        """Create from tuple"""
        return cls(t[0], t[1], t[2])

    @classmethod
    def from_list(cls, l: List[float]) -> "Vector3D":
        """Create from list"""
        return cls(l[0], l[1], l[2])


class Matrix3x3:
    """3x3 Matrix class for transformations"""

    def __init__(self, elements: Optional[List[List[float]]] = None):
        if elements is None:
            self.elements = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        else:
            self.elements = elements

    def __mul__(
        self, other: Union["Matrix3x3", Vector3D, float]
    ) -> Union["Matrix3x3", Vector3D]:
        if isinstance(other, Matrix3x3):
            return self._multiply_matrix(other)
        elif isinstance(other, Vector3D):
            return self._multiply_vector(other)
        elif isinstance(other, (int, float)):
            return self._multiply_scalar(other)
        else:
            raise TypeError("Unsupported multiplication type")

    def _multiply_matrix(self, other: "Matrix3x3") -> "Matrix3x3":
        result = [[0.0, 0.0, 0.0] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    result[i][j] += self.elements[i][k] * other.elements[k][j]
        return Matrix3x3(result)

    def _multiply_vector(self, vector: Vector3D) -> Vector3D:
        x = (
            self.elements[0][0] * vector.x
            + self.elements[0][1] * vector.y
            + self.elements[0][2] * vector.z
        )
        y = (
            self.elements[1][0] * vector.x
            + self.elements[1][1] * vector.y
            + self.elements[1][2] * vector.z
        )
        z = (
            self.elements[2][0] * vector.x
            + self.elements[2][1] * vector.y
            + self.elements[2][2] * vector.z
        )
        return Vector3D(x, y, z)

    def _multiply_scalar(self, scalar: float) -> "Matrix3x3":
        result = [[self.elements[i][j] * scalar for j in range(3)] for i in range(3)]
        return Matrix3x3(result)

    def transpose(self) -> "Matrix3x3":
        """Return transpose of matrix"""
        return Matrix3x3(
            [
                [self.elements[0][0], self.elements[1][0], self.elements[2][0]],
                [self.elements[0][1], self.elements[1][1], self.elements[2][1]],
                [self.elements[0][2], self.elements[1][2], self.elements[2][2]],
            ]
        )

    def determinant(self) -> float:
        """Calculate matrix determinant"""
        a, b, c = self.elements[0]
        d, e, f = self.elements[1]
        g, h, i = self.elements[2]
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

    def inverse(self) -> "Matrix3x3":
        """Return inverse of matrix"""
        det = self.determinant()
        if abs(det) < 1e-10:
            raise ValueError("Matrix is singular")

        a, b, c = self.elements[0]
        d, e, f = self.elements[1]
        g, h, i = self.elements[2]

        adjugate = Matrix3x3(
            [
                [e * i - f * h, c * h - b * i, b * f - c * e],
                [f * g - d * i, a * i - c * g, c * d - a * f],
                [d * h - e * g, b * g - a * h, a * e - b * d],
            ]
        )

        return adjugate * (1.0 / det)


def calculate_distance(
    point1: Union[Vector3D, Tuple[float, float, float], List[float]],
    point2: Union[Vector3D, Tuple[float, float, float], List[float]],
) -> float:
    """Calculate Euclidean distance between two points"""
    if isinstance(point1, Vector3D):
        p1 = point1.to_tuple()
    else:
        p1 = tuple(point1)

    if isinstance(point2, Vector3D):
        p2 = point2.to_tuple()
    else:
        p2 = tuple(point2)

    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def calculate_angle(point1: Vector3D, point2: Vector3D, point3: Vector3D) -> float:
    """Calculate angle between three points (point2 is vertex)"""
    v1 = point1 - point2
    v2 = point3 - point2

    cos_angle = v1.dot(v2) / (v1.magnitude() * v2.magnitude())
    cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to avoid numerical errors

    return math.acos(cos_angle)


def calculate_dihedral_angle(
    point1: Vector3D, point2: Vector3D, point3: Vector3D, point4: Vector3D
) -> float:
    """Calculate dihedral angle between four points"""
    v1 = point2 - point1
    v2 = point3 - point2
    v3 = point4 - point3

    n1 = v1.cross(v2)
    n2 = v2.cross(v3)

    cos_angle = n1.dot(n2) / (n1.magnitude() * n2.magnitude())
    cos_angle = max(-1.0, min(1.0, cos_angle))

    angle = math.acos(cos_angle)

    sign = 1 if v2.dot(n1.cross(n2)) > 0 else -1

    return sign * angle


def create_sphere_mesh(radius: float, resolution: int = 12) -> Dict[str, Any]:
    """Create sphere mesh data for rendering"""
    vertices = []
    faces = []

    for i in range(resolution + 1):
        theta = math.pi * i / resolution
        for j in range(resolution + 1):
            phi = 2 * math.pi * j / resolution

            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.sin(theta) * math.sin(phi)
            z = radius * math.cos(theta)

            vertices.append([x, y, z])

    for i in range(resolution):
        for j in range(resolution):
            first = i * (resolution + 1) + j
            second = first + resolution + 1

            faces.extend([[first, second, first + 1], [second, second + 1, first + 1]])

    return {
        "vertices": vertices,
        "faces": faces,
        "radius": radius,
        "resolution": resolution,
    }


def create_cylinder_mesh(
    radius: float, height: float, resolution: int = 8
) -> Dict[str, Any]:
    """Create cylinder mesh data for bonds"""
    vertices = []
    faces = []

    for i in range(resolution):
        angle = 2 * math.pi * i / resolution
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)

        vertices.append([x, y, 0])
        vertices.append([x, y, height])

    for i in range(resolution):
        next_i = (i + 1) % resolution

        bottom1 = i * 2
        bottom2 = next_i * 2
        top1 = i * 2 + 1
        top2 = next_i * 2 + 1

        faces.extend([[bottom1, bottom2, top2], [bottom1, top2, top1]])

    return {
        "vertices": vertices,
        "faces": faces,
        "radius": radius,
        "height": height,
        "resolution": resolution,
    }


def transform_coordinates(
    coordinates: List[List[float]],
    transformation_matrix: Matrix3x3,
    translation: Vector3D = None,
) -> List[List[float]]:
    """Transform coordinates using matrix and translation"""
    transformed = []

    for coord in coordinates:
        point = Vector3D(coord[0], coord[1], coord[2])
        transformed_point = transformation_matrix * point

        if translation:
            transformed_point = transformed_point + translation

        transformed.append(transformed_point.to_list())

    return transformed


def calculate_bounding_box(coordinates: List[List[float]]) -> Tuple[Vector3D, Vector3D]:
    """Calculate bounding box of coordinates"""
    if not coordinates:
        return Vector3D(0, 0, 0), Vector3D(0, 0, 0)

    min_coords = [min(c[i] for c in coordinates) for i in range(3)]
    max_coords = [max(c[i] for c in coordinates) for i in range(3)]

    return Vector3D(*min_coords), Vector3D(*max_coords)


def calculate_center_of_mass(
    coordinates: List[List[float]], masses: Optional[List[float]] = None
) -> Vector3D:
    """Calculate center of mass"""
    if not coordinates:
        return Vector3D(0, 0, 0)

    if masses is None:
        masses = [1.0] * len(coordinates)

    total_mass = sum(masses)
    if total_mass == 0:
        return Vector3D(0, 0, 0)

    com_x = sum(c[0] * m for c, m in zip(coordinates, masses)) / total_mass
    com_y = sum(c[1] * m for c, m in zip(coordinates, masses)) / total_mass
    com_z = sum(c[2] * m for c, m in zip(coordinates, masses)) / total_mass

    return Vector3D(com_x, com_y, com_z)


def find_optimal_camera_position(
    center: Vector3D, radius: float, camera_distance_multiplier: float = 2.0
) -> Vector3D:
    """Find optimal camera position for viewing"""
    direction = Vector3D(1, 1, 1).normalize()
    distance = radius * camera_distance_multiplier

    return center + direction * distance


@lru_cache(maxsize=100)
def get_atomic_radius_cached(element: str, scale: float = 1.0) -> float:
    """Cached atomic radius lookup"""
    return get_atomic_radius(element, scale)


def optimize_mesh_resolution(num_atoms: int, target_fps: float = 30.0) -> int:
    """Determine optimal mesh resolution based on atom count and performance target"""
    if num_atoms < 100:
        return 24  # High quality
    elif num_atoms < 500:
        return 16  # Medium-high quality
    elif num_atoms < 2000:
        return 12  # Medium quality
    elif num_atoms < 10000:
        return 8  # Low-medium quality
    else:
        return 6  # Low quality for performance


def calculate_optimal_lod_distance(view_distance: float, object_size: float) -> float:
    """Calculate optimal level-of-detail distance"""
    return view_distance * object_size * 0.01


def create_coordinate_system_arrows(length: float = 1.0) -> Dict[str, Any]:
    """Create coordinate system arrows for visualization"""
    arrows = {}

    # X-axis (red)
    arrows["x"] = {
        "start": [0, 0, 0],
        "end": [length, 0, 0],
        "color": [1.0, 0.0, 0.0],
        "label": "X",
    }

    # Y-axis (green)
    arrows["y"] = {
        "start": [0, 0, 0],
        "end": [0, length, 0],
        "color": [0.0, 1.0, 0.0],
        "label": "Y",
    }

    # Z-axis (blue)
    arrows["z"] = {
        "start": [0, 0, 0],
        "end": [0, 0, length],
        "color": [0.0, 0.0, 1.0],
        "label": "Z",
    }

    return arrows
