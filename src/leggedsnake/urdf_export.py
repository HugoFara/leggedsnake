"""
URDF export for walking linkages.

Converts a :class:`Walker` (hypergraph topology + dimensions) into a
`URDF <http://wiki.ros.org/urdf>`_ XML string or file, suitable for
visualization and simulation in ROS, Gazebo, RViz, MuJoCo, and other
robotics tools.

Mapping from leggedsnake concepts to URDF:

* **URDF link** — each edge (binary rigid bar) or hyperedge (rigid body
  group) becomes a URDF ``<link>`` with a cylinder visual/collision and
  computed inertial properties.
* **URDF joint** — each non-ground node becomes a revolute ``<joint>``
  connecting two URDF links.  Ground nodes produce ``fixed`` joints
  attaching to the base frame.  Driver nodes produce ``continuous``
  (actuated) joints.
* **Base link** — the ground/chassis frame.  All ground nodes are fixed
  to this link.

Example::

    from leggedsnake import Walker
    from leggedsnake.urdf_export import to_urdf

    walker = ...  # build or load a Walker
    urdf_string = to_urdf(walker)
    with open("walker.urdf", "w") as f:
        f.write(urdf_string)
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .walker import Walker


@dataclass
class URDFConfig:
    """Parameters controlling URDF generation.

    Attributes
    ----------
    link_radius : float
        Radius of cylinder geometry for each link (m).
    density : float
        Material density for inertia computation (kg/m^3).
    base_link_name : str
        Name of the root/chassis URDF link.
    effort_limit : float
        Maximum joint effort for actuated (driver) joints (N*m).
    velocity_limit : float
        Maximum joint velocity for actuated joints (rad/s).
    mesh_color : tuple[float, float, float, float]
        RGBA color for visual elements.
    """

    link_radius: float = 0.02
    density: float = 1000.0
    base_link_name: str = "base_link"
    effort_limit: float = 1000.0
    velocity_limit: float = 10.0
    mesh_color: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)


def to_urdf(
    walker: Walker,
    config: URDFConfig | None = None,
) -> str:
    """Convert a Walker to a URDF XML string.

    Parameters
    ----------
    walker : Walker
        The walking linkage to export.
    config : URDFConfig | None
        Export parameters. Uses defaults if None.

    Returns
    -------
    str
        A complete URDF XML document as a string.
    """
    if config is None:
        config = URDFConfig()

    robot = ET.Element("robot", name=walker.name or "walker")

    # Collect topology data
    topo = walker.topology
    dims = walker.dimensions

    ground_ids = {n.id for n in topo.ground_nodes()}
    driver_ids = {n.id for n in topo.driver_nodes()}

    # --- Build edge-to-URDF-link mapping ---
    # Each edge becomes a URDF link (a rigid bar between two nodes).
    # Hyperedge members are merged: all edges within a hyperedge share
    # one URDF link.

    # Map: edge_id -> urdf_link_name
    edge_to_urdf_link: dict[str, str] = {}
    # Map: urdf_link_name -> list of (source_node, target_node, length)
    urdf_links: dict[str, list[tuple[str, str, float]]] = {}

    # First, group edges by hyperedge membership
    edge_to_hyperedge: dict[str, str] = {}
    for he_id, he in topo.hyperedges.items():
        for e in he.to_edges():
            edge_to_hyperedge[e.id] = he_id
            # Also check existing edges that connect the same node pair
            for eid, edge in topo.edges.items():
                if {edge.source, edge.target} <= set(he.nodes):
                    edge_to_hyperedge[eid] = he_id

    # Assign URDF link names
    assigned_hyperedges: set[str] = set()
    for eid, edge in topo.edges.items():
        he_id = edge_to_hyperedge.get(eid)
        if he_id is not None and he_id not in assigned_hyperedges:
            # First edge in this hyperedge group
            link_name = _sanitize(f"link_{he_id}")
            assigned_hyperedges.add(he_id)
            # Add all edges of this hyperedge
            for eid2, e2 in topo.edges.items():
                if edge_to_hyperedge.get(eid2) == he_id:
                    edge_to_urdf_link[eid2] = link_name
                    dist = dims.get_edge_distance(eid2) or _node_distance(
                        dims, e2.source, e2.target
                    )
                    if link_name not in urdf_links:
                        urdf_links[link_name] = []
                    urdf_links[link_name].append((e2.source, e2.target, dist))
        elif he_id is not None:
            # Already assigned
            link_name = _sanitize(f"link_{he_id}")
            edge_to_urdf_link[eid] = link_name
        else:
            # Standalone edge
            link_name = _sanitize(f"link_{eid}")
            edge_to_urdf_link[eid] = link_name
            dist = dims.get_edge_distance(eid) or _node_distance(
                dims, edge.source, edge.target
            )
            urdf_links[link_name] = [(edge.source, edge.target, dist)]

    # --- Create base link ---
    _add_link_element(robot, config.base_link_name, config, segments=[])

    # --- Create URDF links ---
    for link_name, segments in urdf_links.items():
        _add_link_element(robot, link_name, config, segments=segments)

    # --- Create URDF joints ---
    # Strategy: for each node, determine which URDF links it connects.
    # A node connects all edges incident on it. We create joints between
    # adjacent URDF links at each node.

    joint_counter = 0
    created_joint_pairs: set[tuple[str, str]] = set()

    for node_id, node in topo.nodes.items():
        pos = dims.get_node_position(node_id)
        if pos is None:
            pos = (0.0, 0.0)

        # Find all URDF links incident on this node
        incident_edges = topo.get_edges_for_node(node_id)
        incident_urdf_links = list(dict.fromkeys(
            edge_to_urdf_link[e.id]
            for e in incident_edges
            if e.id in edge_to_urdf_link
        ))

        if node_id in ground_ids:
            # Ground node: fix all incident links to base
            for ulink in incident_urdf_links:
                pair = (config.base_link_name, ulink)
                if pair in created_joint_pairs:
                    continue
                created_joint_pairs.add(pair)
                _add_joint_element(
                    robot,
                    name=_sanitize(f"ground_{node_id}_{ulink}"),
                    joint_type="fixed",
                    parent=config.base_link_name,
                    child=ulink,
                    origin_xyz=(pos[0], pos[1], 0.0),
                )
                joint_counter += 1

        elif node_id in driver_ids:
            # Driver node: first new joint is continuous (actuated),
            # subsequent joints at this node are revolute.
            if incident_urdf_links:
                parent_link = config.base_link_name
                first_new = True
                for i, ulink in enumerate(incident_urdf_links):
                    pair = (parent_link, ulink)
                    if pair in created_joint_pairs:
                        parent_link = ulink
                        continue
                    created_joint_pairs.add(pair)
                    jtype = "continuous" if first_new else "revolute"
                    first_new = False
                    _add_joint_element(
                        robot,
                        name=_sanitize(f"motor_{node_id}_{i}"),
                        joint_type=jtype,
                        parent=parent_link,
                        child=ulink,
                        origin_xyz=(pos[0], pos[1], 0.0),
                        axis=(0, 0, 1),
                        effort=config.effort_limit,
                        velocity=config.velocity_limit,
                    )
                    parent_link = ulink
                    joint_counter += 1

        else:
            # Driven node: revolute joints chaining incident links
            if len(incident_urdf_links) >= 2:
                for i in range(1, len(incident_urdf_links)):
                    parent_link = incident_urdf_links[i - 1]
                    child_link = incident_urdf_links[i]
                    pair_key = tuple(sorted((parent_link, child_link)))
                    if pair_key in created_joint_pairs:
                        continue
                    created_joint_pairs.add(pair_key)
                    _add_joint_element(
                        robot,
                        name=_sanitize(f"joint_{node_id}_{i}"),
                        joint_type="revolute",
                        parent=parent_link,
                        child=child_link,
                        origin_xyz=(pos[0], pos[1], 0.0),
                        axis=(0, 0, 1),
                        limit_lower=-math.pi,
                        limit_upper=math.pi,
                    )
                    joint_counter += 1

    # Pretty-print
    ET.indent(robot, space="  ")
    return ET.tostring(robot, encoding="unicode", xml_declaration=True)


def to_urdf_file(
    walker: Walker,
    path: str,
    config: URDFConfig | None = None,
) -> None:
    """Write a Walker as a URDF file.

    Parameters
    ----------
    walker : Walker
        The walking linkage to export.
    path : str
        Output file path.
    config : URDFConfig | None
        Export parameters.
    """
    urdf_str = to_urdf(walker, config=config)
    with open(path, "w") as f:
        f.write(urdf_str)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sanitize(name: str) -> str:
    """Make a string safe for URDF element names."""
    return name.replace(" ", "_").replace("(", "").replace(")", "")


def _node_distance(
    dims: object, src: str, tgt: str,
) -> float:
    """Compute distance between two nodes from their positions."""
    from pylinkage.dimensions import Dimensions

    if not isinstance(dims, Dimensions):
        return 0.1  # fallback
    p1 = dims.get_node_position(src)
    p2 = dims.get_node_position(tgt)
    if p1 is None or p2 is None:
        return 0.1
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _cylinder_inertia(
    mass: float, radius: float, length: float,
) -> tuple[float, float, float]:
    """Approximate inertia of a thin cylinder along Z axis.

    Returns (ixx, iyy, izz) for a cylinder aligned with Y axis.
    """
    ixx = mass * (3 * radius**2 + length**2) / 12
    iyy = mass * radius**2 / 2
    izz = ixx  # symmetric
    return ixx, iyy, izz


def _add_link_element(
    parent: ET.Element,
    name: str,
    config: URDFConfig,
    segments: list[tuple[str, str, float]],
) -> ET.Element:
    """Add a <link> element to the URDF tree.

    For the base link (no segments), creates a minimal link with zero inertia.
    For bar links, creates cylinder geometry and computed inertial properties.
    """
    link = ET.SubElement(parent, "link", name=name)

    if not segments:
        # Base link: minimal inertial
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", value="1.0")
        ET.SubElement(
            inertial, "inertia",
            ixx="0.001", iyy="0.001", izz="0.001",
            ixy="0", ixz="0", iyz="0",
        )
        return link

    # Use the longest segment for the primary geometry
    total_length = sum(s[2] for s in segments)
    avg_length = total_length / len(segments) if segments else 0.1
    r = config.link_radius

    # Inertial
    volume = math.pi * r**2 * total_length
    mass = config.density * volume
    ixx, iyy, izz = _cylinder_inertia(mass, r, avg_length)

    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", value=f"{mass:.6f}")
    ET.SubElement(
        inertial, "inertia",
        ixx=f"{ixx:.6f}", iyy=f"{iyy:.6f}", izz=f"{izz:.6f}",
        ixy="0", ixz="0", iyz="0",
    )

    # Visual: one cylinder per segment
    r_val, g_val, b_val, a_val = config.mesh_color
    for i, (src, tgt, length) in enumerate(segments):
        visual = ET.SubElement(link, "visual")
        geom = ET.SubElement(visual, "geometry")
        ET.SubElement(geom, "cylinder", radius=f"{r:.4f}", length=f"{length:.4f}")

        # Origin at segment midpoint, rotated to align with Y
        ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")

        material = ET.SubElement(visual, "material", name=f"mat_{name}_{i}")
        ET.SubElement(
            material, "color",
            rgba=f"{r_val} {g_val} {b_val} {a_val}",
        )

    # Collision: single cylinder for simplicity
    collision = ET.SubElement(link, "collision")
    cgeom = ET.SubElement(collision, "geometry")
    ET.SubElement(
        cgeom, "cylinder",
        radius=f"{r:.4f}", length=f"{avg_length:.4f}",
    )

    return link


def _add_joint_element(
    xml_parent: ET.Element,
    name: str,
    joint_type: str,
    parent: str,  # URDF parent link name
    child: str,
    origin_xyz: tuple[float, float, float] = (0, 0, 0),
    axis: tuple[int, int, int] | None = None,
    limit_lower: float | None = None,
    limit_upper: float | None = None,
    effort: float | None = None,
    velocity: float | None = None,
) -> ET.Element:
    """Add a <joint> element to the URDF tree."""
    joint = ET.SubElement(xml_parent, "joint", name=name, type=joint_type)
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    ET.SubElement(
        joint, "origin",
        xyz=f"{origin_xyz[0]:.6f} {origin_xyz[1]:.6f} {origin_xyz[2]:.6f}",
        rpy="0 0 0",
    )

    if axis is not None:
        ET.SubElement(joint, "axis", xyz=f"{axis[0]} {axis[1]} {axis[2]}")

    if joint_type in ("revolute", "prismatic"):
        limit_attrs: dict[str, str] = {}
        if limit_lower is not None:
            limit_attrs["lower"] = f"{limit_lower:.4f}"
        if limit_upper is not None:
            limit_attrs["upper"] = f"{limit_upper:.4f}"
        if effort is not None:
            limit_attrs["effort"] = f"{effort:.1f}"
        else:
            limit_attrs["effort"] = "1000.0"
        if velocity is not None:
            limit_attrs["velocity"] = f"{velocity:.1f}"
        else:
            limit_attrs["velocity"] = "10.0"
        ET.SubElement(joint, "limit", **limit_attrs)

    if joint_type == "continuous" and (effort is not None or velocity is not None):
        limit_attrs = {}
        if effort is not None:
            limit_attrs["effort"] = f"{effort:.1f}"
        if velocity is not None:
            limit_attrs["velocity"] = f"{velocity:.1f}"
        if limit_attrs:
            ET.SubElement(joint, "limit", **limit_attrs)

    return joint
