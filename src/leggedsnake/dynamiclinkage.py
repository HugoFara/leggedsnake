#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The dynamiclinkage module is an interface between Pymunk and a kinematic linkage.

It provides various classes to overwrite the pyJoint objects.
It also includes a DynamicLinkage that inherits from
pylinkage.Linkage.
Eventually a handy convert_to_dynamic_linkage method can generate a
DynamicLinkage from a pylinkage.Linkage.
"""
from __future__ import annotations

import abc
import warnings
from math import atan2
from typing import TYPE_CHECKING, Any

import pymunk as pm

# Legacy joint classes required for DynamicJoint inheritance (Nail, PinUp,
# DynamicPivot, Motor all extend these).  Suppress pylinkage 0.8.0 deprecation.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, message=r"pylinkage\.joints"
    )
    from pylinkage import (
        Crank, Fixed, Linkage, Pivot, Static, UnbuildableError
    )
    from pylinkage.joints.joint import Joint

from pylinkage.components import Ground
from pylinkage.dimensions import Dimensions
from pylinkage.dyads import FixedDyad, RRRDyad
from pylinkage.geometry import cyl_to_cart
from pylinkage.geometry.core import dist
from pylinkage.hypergraph import NodeRole, from_linkage

from .hypergraph_physics import (
    PhysicsMapping,
    create_bodies_from_hypergraph,
)

if TYPE_CHECKING:
    from pylinkage.hypergraph import HypergraphLinkage


class DynamicJoint(abc.ABC):
    """Dynamic, pymunk compatible equivalent of kinematic Joint."""

    _a: pm.Body
    _b: pm.Body
    _anchor_a: pm.Vec2d
    _anchor_b: pm.Vec2d
    space: pm.Space | None
    radius: float
    density: float
    filter: pm.ShapeFilter | None
    superposed: set[Any]

    def __init__(
        self,
        body0: pm.Body | None = None,
        body1: pm.Body | None = None,
        space: pm.Space | None = None,
        radius: float = 0.3,
        density: float = 1,
        shape_filter: pm.ShapeFilter | None = None,
    ) -> None:
        """
        Partial class to generate a DynamicJoint.

        A DynamicJoint is a Joint that can handle Pymunk Bodies.

        Parameters
        ----------
        body0 : pymunk.body.Body, optional
            First Body to link to. The default is None.
        body1 : pymunk.body.Body, optional
            Second Body to link to. The default is None
        space : pymunk.space.Space, optional
            Simulation space in which the DynamicJoint exists.
            The default is None.
        radius : float, optional
            DynamicJoint will generate hulls of this radius. The default is .3.
        density : float, optional
            Density of the hull, mass will be computed accordingly.
            The default is 1.
        shape_filter : pymunk.shapes.ShapeFilter, optional
            Prevent hulls from colliding with each other. Useful is the same
            linkage for instance. The default is None.
        """
        if isinstance(body0, pm.Body):
            self._a = body0
            self._anchor_a = body0.world_to_local(self.coord())
        if isinstance(body1, pm.Body):
            self._b = body1
            self._anchor_b = body1.world_to_local(self.coord())
        self.space = space
        self.radius = radius
        self.density = density
        self.filter = shape_filter
        # All the Joint/constraint that are on the same bar
        self.superposed = set()

    @property
    @abc.abstractmethod
    def x(self) -> float:
        """Return the x coordinate."""
        ...

    @property
    @abc.abstractmethod
    def y(self) -> float:
        """Return the y coordinate."""
        ...

    @abc.abstractmethod
    def coord(self) -> pm.Vec2d:
        """Return the coordinates of this joint."""
        ...

    @abc.abstractmethod
    def set_coord(self, *args: float) -> None:
        """Set the coordinates of this joint."""
        ...

    def __generate_link__(self, body: pm.Body, parent_pos: pm.Vec2d) -> pm.Segment:
        """
        Create a pymunk.Segment between two joints on a body and return it.

        The first extremity of the segment is self Joint position, the other is
        parent_pos (a position).

        Parameters
        ----------
        body : pymunk.Body
            The Body you want the link to be added to.
        parent_pos : sequence
            Any sequence of two floats. It give th

        Returns
        -------
        seg : pymunk.Segment
            The segment joining the two joints of the body.

        """
        seg = pm.Segment(
            body, body.world_to_local(self.coord()),
            body.world_to_local(parent_pos),
            self.radius)
        seg.density = self.density
        if self.filter is not None:
            seg.filter = self.filter
        return seg

    def __generate_body__(self, index: int = 0) -> None:
        """
        Create a pymunk.Body and assign it to the specified interface.

        Parameters
        ----------
        index : int, optional
            The interface index to assign the body to.
            0: assign to self.a
            1: assign to self.b
            2: assign to self.a and self.b
            The default is 0.

        Returns
        -------
        None.

        """
        if index == 2:
            self.__generate_body__(0)
            self.__generate_body__(1)
            return
        sindex = str(index)
        if (
                hasattr(self, 'joint' + sindex)
                and isinstance(getattr(self, 'joint' + sindex), Joint)
        ):
            body = pm.Body()
            body.mass = 1
            parent_pos = pm.Vec2d(*getattr(self, 'joint' + sindex).coord())
            body.position = (parent_pos + self.coord()) / 2
            seg = self.__generate_link__(body, parent_pos)
            if index == 0:
                self._a = body
                self._anchor_a = body.world_to_local(self.coord())
            else:
                self._b = body
                self._anchor_b = body.world_to_local(self.coord())
            assert self.space is not None
            self.space.add(body, seg)

    def __find_common_body__(self) -> pm.Body:
        """Find the body linking two joints."""
        joint0: DynamicJoint = getattr(self, 'joint0')
        joint1: DynamicJoint = getattr(self, 'joint1')
        joints = joint0, joint1
        bodies: list[set[pm.Body]] = [set((joints[0]._a,)), set((joints[1]._a,))]
        for body_set, j in zip(bodies, joints):
            body_set.add(j._b)
            for sup in j.superposed:
                body_set.add(sup._a)
                body_set.add(sup._b)
        inter = bodies[0].intersection(bodies[1])
        if len(inter) == 0:
            message = ('Unable to find a common body between parents joints'
                       ' {} and {}.').format(*joints)
            raise UnbuildableError(self, message=message)
        if len(inter) == 1:
            return inter.pop()
        else:
            message = ('Multiple common bodies between parents joints'
                       ' {} and {}, parents of {}: {}.')
            raise Exception(message.format(joints[0], joints[1], self, inter))

    def reload(self) -> None:
        """
        Reload DynamicJoint coordinates.

        Coordinates only depend on Body position and DO NOT use any linkage
        constraint.

        Returns
        -------
        None.

        """
        if hasattr(self, '_b') and hasattr(self, '_anchor_b'):
            self.set_coord(*self._b.local_to_world(self._anchor_b))


class Nail(Static, DynamicJoint):  # type: ignore[misc]
    """
    A simple point to follow a rigidbody.

    It is special since it DOES NOT generate bodies.
    """

    _distance0: float
    _angle0: float

    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        name: str | None = None,
        body: pm.Body | None = None,
        space: pm.Space | None = None,
        radius: float = 0.3,
        density: float = 1,
        shape_filter: pm.ShapeFilter | None = None,
    ) -> None:
        Static.__init__(self, x, y, name=name)
        DynamicJoint.__init__(
            self, body0=body, body1=body, space=space, radius=radius,
            density=density, shape_filter=shape_filter)
        if body is not None:
            self._a = self._b = body
        self.space = space
        self.radius = radius
        self.density = density
        self.filter = shape_filter
        if hasattr(self, '_a') and self._a is not None:
            self.__set_offset__()
            if hasattr(self, 'joint0') and isinstance(self.joint0, Joint):
                seg = pm.Segment(self._a, self._a.world_to_local(self.coord()),
                                 self._a.world_to_local(self.joint0.coord()),
                                 self.radius)
                seg.density = self.density
                if self.filter is not None:
                    seg.filter = self.filter
                assert self.space is not None
                self.space.add(seg)
            if hasattr(self, 'joint1') and isinstance(self.joint1, Joint):
                seg = pm.Segment(self._a, self._a.world_to_local(self.coord()),
                                 self._a.world_to_local(self.joint1.coord()),
                                 self.radius)
                seg.density = self.density
                if self.filter is not None:
                    seg.filter = self.filter
                assert self.space is not None
                self.space.add(seg)

    def __set_offset__(self) -> None:
        """Memorize the offset distance between self and linked Body."""
        self._distance0 = dist(*self.coord(), *self._a.position)
        x_pos, y_pos = self.coord() - self._a.position
        self._angle0 = atan2(y_pos, x_pos) - self._a.angle

    def reload(self) -> None:
        """Reload position based on linked body rotation and position."""
        # Unpack Vec2d position for cyl_to_cart (takes ori_x, ori_y separately)
        pos = self._a.position
        self.set_coord(cyl_to_cart(self._distance0,
                                   self._a.angle + self._angle0,
                                   pos.x, pos.y))


class PinUp(Fixed, DynamicJoint):  # type: ignore[misc]
    """
    Dynamic counterpart of Fixed joint.

    Add two pm.Segment to the linked body.
    """

    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        joint0: DynamicJoint | None = None,
        space: pm.Space | None = None,
        joint1: DynamicJoint | None = None,
        distance: float | None = None,
        angle: float | None = None,
        name: str | None = None,
        radius: float = 0.3,
        density: float = 1,
        shape_filter: pm.ShapeFilter | None = None,
    ) -> None:
        Fixed.__init__(
            self, x, y, joint0=joint0, joint1=joint1, name=name,
            distance=distance, angle=angle)
        DynamicJoint.__init__(
            self, space=space, radius=radius, density=density,
            shape_filter=shape_filter)
        if (
                isinstance(self.joint0, Joint) and
                isinstance(self.joint1, Joint)
        ):
            Fixed.reload(self)
        if self.joint0 is not None:
            self.set_anchor_a(self.joint0, self.r, self.angle)
        if self.joint1 is not None:
            self.set_anchor_b(self.joint1)

    def set_anchor_a(
        self,
        joint: DynamicJoint,
        distance: float | None = None,
        angle: float | None = None,
    ) -> None:
        """
        Set first anchor characteristics.

        Parameters
        ----------
        joint : DynamicJoint
            DynamicJoint to use as anchor_a.
        distance : float, optional
            Distance to keep constant. The default is None.
        angle : float, optional
            Angle (in radians)  (joint1, joint0, self). The default is None.

        Returns
        -------
        None.

        """
        Fixed.set_anchor0(self, joint, distance, angle)

    def set_anchor_b(self, joint: DynamicJoint) -> None:
        """
        Set second anchor characteristics.

        It will create errors if called before anchor_a is properly defined.
        SIDE EFFECT: creates two Segment objects and adds them to self.space.

        Parameters
        ----------
        joint : DynamicJoint
            Joint to use as anchor_b.

        Returns
        -------
        None.

        """
        Fixed.set_anchor1(self, joint)
        Fixed.reload(self)
        self._b = self._a = super().__find_common_body__()
        self._anchor_a = self._a.world_to_local(self.coord())
        assert self.space is not None
        self.space.add(self.__generate_link__(self._a, joint.coord()))

        self._anchor_b = self._b.world_to_local(self.coord())
        self.space.add(self.__generate_link__(self._a, joint.coord()))

    def reload(self) -> None:
        """
        Reload DynamicJoint coordinates.

        Coordinates only depend on Body position and DO NOT use any linkage
        constraint.

        Returns
        -------
        None.

        """
        DynamicJoint.reload(self)


class DynamicPivot(Pivot, DynamicJoint):  # type: ignore[misc]
    """Dynamic counterpart of a Pivot joint."""

    pivot: pm.PivotJoint

    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        joint0: DynamicJoint | None = None,
        space: pm.Space | None = None,
        joint1: DynamicJoint | None = None,
        distance0: float | None = None,
        distance1: float | None = None,
        name: str | None = None,
        radius: float = 0.3,
        density: float = 1,
        shape_filter: pm.ShapeFilter | None = None,
    ) -> None:
        Pivot.__init__(
            self, x, y,
            joint0=joint0, joint1=joint1,
            name=name,
            distance0=distance0, distance1=distance1
        )
        DynamicJoint.__init__(
            self, space=space, radius=radius, density=density,
            shape_filter=shape_filter
        )
        if self.joint0 is not None:
            self.set_anchor_a(self.joint0, self.r0)
        if self.joint1 is not None:
            self.set_anchor_b(self.joint1, self.r1)

    def set_anchor_a(
        self, joint: DynamicJoint, distance: float | None = None
    ) -> None:
        """
        Set anchor_a characteristics.

        Parameters
        ----------
        joint : DynamicJoint
            DynamicJoint to use as anchor_a.
        distance : float, optional
            Distance to keep constant between anchor_a and self.
            The default is None.

        Returns
        -------
        None.

        """
        Pivot.set_anchor0(self, joint, distance)
        if not hasattr(self, '_a'):
            self.__generate_body__(0)
            if isinstance(joint, Nail):
                parent_body0 = joint._a
            else:
                parent_body0 = joint._b
            pivot = pm.PivotJoint(
                self._a, parent_body0,
                joint.coord())
            pivot.collide_bodies = False
            joint.superposed.add(pivot)
            assert self.space is not None
            self.space.add(pivot)

    def set_anchor_b(
        self, joint: DynamicJoint, distance: float | None = None
    ) -> None:
        """
        Set anchor_b characteristics.

        Parameters
        ----------
        joint : DynamicJoint
            DynamicJoint to use as anchor_b.
        distance : float, optional
            Distance to keep constant between anchor_b and self.
            The default is None.

        Returns
        -------
        None.

        """
        Pivot.set_anchor1(self, joint, distance)
        if not hasattr(self, '_b'):
            self.__generate_body__(1)
            # PivotJoint on self position
            self.pivot = pm.PivotJoint(self._a, self._b, self.coord())
            if isinstance(joint, Nail):
                parent_body1 = joint._a
            else:
                parent_body1 = joint._b
            pivot_b = pm.PivotJoint(self._b, parent_body1, joint.coord())
            joint.superposed.add(pivot_b)
            assert self.space is not None
            self.space.add(self.pivot, pivot_b)

    def reload(self) -> None:
        """
        Reload DynamicJoint coordinates.

        Coordinates only depend on Body position and DO NOT use any linkage
        constraint.

        Returns
        -------
        None.

        """
        DynamicJoint.reload(self)


class Motor(Crank, DynamicJoint):  # type: ignore[misc]
    """
    A Motor is a crank.

    It makes a link between a body and a second link it creates. It attaches
    them with a PivotJoint, and adds a SimpleMotor over it. The bodies are now
    constrained to rotate around one each other.

    The Motor is placed at the extremity of the body it creates.
    """

    pivot: pm.PivotJoint
    actuator: pm.SimpleMotor

    def __init__(
        self,
        x: float | None = None,
        y: float | None = None,
        joint0: DynamicJoint | None = None,
        space: pm.Space | None = None,
        distance: float | None = None,
        angle: float | None = None,
        name: str | None = None,
        radius: float = 0.3,
        density: float = 1,
        shape_filter: pm.ShapeFilter | None = None,
    ) -> None:
        Crank.__init__(self, x, y, joint0=joint0,
                               distance=distance, angle=angle, name=name)
        Crank.reload(self, dt=0)
        DynamicJoint.__init__(self, space=space, density=density,
                              shape_filter=shape_filter, radius=radius)
        if joint0 is not None:
            self.set_anchor_a(joint0, distance=distance)

    def __get_reference_body__(self, body: pm.Body | None = None) -> pm.Body:
        """
        Find a body that can be used as a reference body.

        Parameters
        ----------
        body : pymunk.Body, optional
            Body to use as reference Body.
            If set to None, use self.joint0_a.
            The default is None.

        Raises
        ------
        ValueError
            Raised when body argument is not None and not in self.joint0
            bodies.

        Returns
        -------
        pymunk.Body
            A valid Body to be used as a reference.

        """
        if body is None:
            ref_body: pm.Body = self.joint0._a
            return ref_body
        if body in (self.joint0._a, self.joint0._b):
            return body
        message = 'Argument body should in {} bodies'
        raise ValueError(message.format(self.joint0))

    def __generate_body__(self) -> None:  # type: ignore[override]
        """Generate the crank body only."""
        if hasattr(self, 'joint0') and isinstance(self.joint0, Joint):
            body = pm.Body()
            body.position = (pm.Vec2d(*self.coord()) + self.joint0.coord()) / 2
            seg = self.__generate_link__(body, self.joint0.coord())
            self._a = self._b = body
            self._anchor_b = body.world_to_local(self.coord())
            assert self.space is not None
            self.space.add(body, seg)

    def set_anchor_a(
        self, joint: DynamicJoint, distance: float | None = None
    ) -> None:
        """
        Set anchor_a characteristics.

        Parameters
        ----------
        joint : DynamicJoint
            DynamicJoint to use as anchor_a.
        distance : float, optional
            Distance to keep constant between anchor_a and self.
            The default is None.

        Returns
        -------
        None.

        """
        Crank.set_anchor0(self, joint, distance)
        ref_body = self.__get_reference_body__()
        if not hasattr(self, '_b'):
            self.__generate_body__()
            self.pivot = pm.PivotJoint(ref_body, self._b, self.joint0.coord())
            self.pivot.collide_bodies = False
            self.actuator = pm.SimpleMotor(self._b, ref_body, self.angle)
            self.joint0.superposed.add(self.pivot)
            self.joint0.superposed.add(self.actuator)
            assert self.space is not None
            self.space.add(self.pivot, self.actuator)

    def reload(self) -> None:
        """
        Reload DynamicJoint coordinates.

        Coordinates only depend on Body position and DO NOT use any linkage
        constraint.

        Returns
        -------
        None.

        """
        DynamicJoint.reload(self)


class DynamicLinkage(Linkage):  # type: ignore[misc]
    """
    Dynamic counterpart of a kinematic Linkage.

    It has several attributes linked to its dynamic nature and is close to an
    empty shell in the ways you will use it.

    Please note that it carries a load, which is relevant with real-world
    simulation where the weight of legs is small compared to the weight of the
    frame.
    """

    __slots__ = [
        'joint_to_rigidbodies', 'rigidbodies', 'body', 'height',
        'mechanical_energy', 'mass', 'space', 'density',
        '_thickness', 'filter', '_cranks', 'joints', '_physics_mapping',
        '_hypergraph', '_dimensions'
    ]

    joint_to_rigidbodies: list[list[pm.Body] | None]
    rigidbodies: list[pm.Body]
    body: pm.Body
    height: float
    mechanical_energy: float
    mass: float
    space: pm.Space
    density: float
    _thickness: float
    filter: pm.ShapeFilter
    _cranks: tuple[Motor, ...]
    joints: tuple[DynamicJoint, ...]
    _physics_mapping: PhysicsMapping
    _hypergraph: HypergraphLinkage
    _dimensions: Dimensions

    def __init__(
        self,
        joints: tuple[Joint, ...],
        space: pm.Space,
        density: float = 1,
        load: float = 0,
        name: str | None = None,
        thickness: float = 0.1,
        motor_rate: float | None = None,
    ) -> None:
        """
        Instanciate a new DynamicLinkage.

        Parameters
        ----------
        joints : Joint
            Joints to be part of the linkage. Kinematic joints will be
            converted to their dynamic equivalents.
        space : pymunk.Space
            Space in which linkage should be instantiated.
        density : float, optional
            Density of the Bodies in the linkage. The default is 1.
        load : float, optional
            Mass of the load to carry. The default is 0.
        name : TYPE, optional
            User-friendly name for the linkage. The default is None. It will
            be set to str(id(self)) if not provided.
        thickness : float, optional
            ratio bar length/width for each bar (radius in pymunk).
            The default is .1.
        motor_rate : float, optional
            Motor angular velocity in rad/s for dynamic simulation.
            If None, falls back to kinematic angle from Crank joints.

        Returns
        -------
        None.

        """
        # Store original joints before super().__init__ may modify them
        original_joints = joints

        super().__init__(joints=joints, name=name)
        self.density = density
        self._thickness = thickness
        self.space = space
        self.filter = pm.ShapeFilter(group=1)

        # Build load body (the frame/chassis)
        load_body = self.build_load(self.joints[0].coord(), load)
        self.body = load_body

        # Convert kinematic linkage to hypergraph representation
        # pylinkage 0.8.0: from_linkage returns (HypergraphLinkage, Dimensions)
        self._hypergraph, self._dimensions = from_linkage(self)

        # Generate physics bodies from hypergraph
        # Pass original joints to detect Fixed joints with static parents
        self._physics_mapping = create_bodies_from_hypergraph(
            self._hypergraph,
            self._dimensions,
            space,
            load_body,
            density,
            thickness,
            self.filter,
            joints=original_joints,
            motor_rate=motor_rate,
        )

        # Create dynamic joint wrappers for coordinate tracking
        self.joints = self._wrap_joints_for_tracking()

        # Collect all rigidbodies
        self.rigidbodies = [load_body]
        for body in self._physics_mapping.edge_to_body.values():
            if body is not load_body and body not in self.rigidbodies:
                self.rigidbodies.append(body)

        # Initialize joint_to_rigidbodies mapping
        self.joint_to_rigidbodies = [None] * len(self.joints)
        for i, joint in enumerate(self.joints):
            if hasattr(joint, 'name') and joint.name:
                bodies = self._physics_mapping.node_to_bodies.get(joint.name)
                if bodies:
                    self.joint_to_rigidbodies[i] = list(bodies)

        self.mass = sum(b.mass for b in self.rigidbodies)
        self._cranks = tuple(j for j in self.joints if isinstance(j, Motor))

    def _wrap_joints_for_tracking(self) -> tuple[DynamicJoint, ...]:
        """Create DynamicJoint wrappers for coordinate tracking.

        Uses the hypergraph and physics mapping to create appropriate
        wrapper joints that track positions from the physics bodies.

        Returns
        -------
        tuple[DynamicJoint, ...]
            Dynamic joint wrappers for all nodes in the hypergraph.
        """
        wrapped_joints: list[DynamicJoint] = []
        hg = self._hypergraph
        dims = self._dimensions
        mapping = self._physics_mapping

        for node_id, node in hg.nodes.items():
            # Get position from dimensions (topology/geometry split in 0.8.0)
            pos = dims.get_node_position(node_id)
            x = pos[0] if pos is not None else 0.0
            y = pos[1] if pos is not None else 0.0
            bodies = mapping.node_to_bodies.get(node_id, set())
            anchors = mapping.node_to_anchors.get(node_id, {})

            # Common kwargs for all joint types
            common = {
                'x': x,
                'y': y,
                'name': node_id,
                'space': self.space,
                'radius': self._thickness,
                'density': self.density,
                'shape_filter': self.filter,
            }

            djoint: DynamicJoint
            if node.role == NodeRole.GROUND:
                # Ground nodes become Nails attached to load body
                djoint = Nail(body=self.body, **common)
            elif node.role == NodeRole.DRIVER:
                # Driver nodes become Motors
                # Find the motor in the physics mapping
                motor_idx = None
                for i, motor in enumerate(mapping.motors):
                    # Check if this motor is associated with this driver
                    if i < len(list(hg.driver_nodes())):
                        driver_list = list(hg.driver_nodes())
                        if i < len(driver_list) and driver_list[i].id == node_id:
                            motor_idx = i
                            break

                driver_angle = dims.get_driver_angle(node_id)
                driver_dist = driver_angle.angular_velocity if driver_angle else 1.0
                driver_ang = driver_angle.angular_velocity if driver_angle else 0.0
                djoint = Motor(
                    joint0=wrapped_joints[0] if wrapped_joints else None,
                    distance=driver_dist,
                    angle=driver_ang,
                    **common,
                )
                # Attach the existing motor and pivot from mapping
                if motor_idx is not None and motor_idx < len(mapping.motors):
                    djoint.actuator = mapping.motors[motor_idx]
                    if motor_idx < len(mapping.motor_pivots):
                        djoint.pivot = mapping.motor_pivots[motor_idx]

                # Set body references
                if bodies:
                    body = next(b for b in bodies if b is not self.body)
                    djoint._a = djoint._b = body
                    if body in anchors:
                        djoint._anchor_a = djoint._anchor_b = anchors[body]
            else:
                # Driven nodes - determine if Pivot or Fixed based on connections
                # For simplicity, create DynamicPivot wrappers
                djoint = DynamicPivot(**common)

                # Set body references from mapping
                if bodies:
                    body_list = list(bodies)
                    if len(body_list) >= 1:
                        djoint._a = body_list[0]
                        if body_list[0] in anchors:
                            djoint._anchor_a = anchors[body_list[0]]
                    if len(body_list) >= 2:
                        djoint._b = body_list[1]
                        if body_list[1] in anchors:
                            djoint._anchor_b = anchors[body_list[1]]
                    elif len(body_list) == 1:
                        djoint._b = body_list[0]
                        djoint._anchor_b = djoint._anchor_a

            # Update joint coordinates from physics bodies
            djoint.reload()
            wrapped_joints.append(djoint)

        return tuple(wrapped_joints)

    def convert_to_dynamic_joints(self, joints: tuple[Joint, ...]) -> None:
        """Convert a kinematic joint in its dynamic counterpart."""
        if not isinstance(self.space, pm.Space):
            raise Exception('Linkage {} Space not defined yet!'.format(self))
        dynajoints: list[DynamicJoint] = []
        conversion_dict: dict[Joint, DynamicJoint] = {}
        common: dict[str, Any] = {
            'space': self.space,
            'radius': self._thickness,
            'density': self.density,
            'shape_filter': self.filter
        }
        for joint in joints:
            common.update({'x': joint.x, 'y': joint.y, 'name': joint.name})
            djoint: DynamicJoint
            if isinstance(joint, DynamicJoint):
                djoint = joint
            elif isinstance(joint, (Static, Ground)):
                djoint = Nail(body=self.body, **common)
            # Joints with at least one reference
            else:
                if isinstance(joint, (Fixed, FixedDyad)):
                    # Legacy: joint0/joint1/r/angle; New: anchor1/anchor2/distance/angle
                    j0 = getattr(joint, 'joint0', None) or getattr(joint, 'anchor1', None)
                    j1 = getattr(joint, 'joint1', None) or getattr(joint, 'anchor2', None)
                    dist_val = getattr(joint, 'r', None) or getattr(joint, 'distance', None)
                    djoint = PinUp(
                        distance=dist_val, angle=joint.angle,
                        joint0=conversion_dict[j0],
                        joint1=conversion_dict[j1],
                        **common
                    )
                elif isinstance(joint, Crank):
                    # Legacy: joint0/r/angle; New (actuators.Crank): anchor/radius/angular_velocity
                    j0 = getattr(joint, 'joint0', None) or getattr(joint, 'anchor', None)
                    dist_val = getattr(joint, 'r', None) or getattr(joint, 'radius', None)
                    ang_val = getattr(joint, 'angle', None) or getattr(joint, 'angular_velocity', None)
                    djoint = Motor(
                        joint0=conversion_dict[j0],
                        distance=dist_val, angle=ang_val,
                        **common
                    )
                elif isinstance(joint, (Pivot, RRRDyad)):
                    # Legacy: joint0/joint1/r0/r1; New: anchor1/anchor2/distance1/distance2
                    j0 = getattr(joint, 'joint0', None) or getattr(joint, 'anchor1', None)
                    j1 = getattr(joint, 'joint1', None) or getattr(joint, 'anchor2', None)
                    d0 = getattr(joint, 'r0', None) or getattr(joint, 'distance1', None)
                    d1 = getattr(joint, 'r1', None) or getattr(joint, 'distance2', None)
                    djoint = DynamicPivot(
                        joint0=conversion_dict[j0],
                        joint1=conversion_dict[j1],
                        distance0=d0, distance1=d1,
                        **common
                    )
                else:
                    continue
            dynajoints.append(djoint)
            conversion_dict[joint] = djoint
        self.joints = tuple(dynajoints)

    def build_load(self, position: pm.Vec2d, load_mass: float) -> pm.Body:
        """Create the load this linkage has to carry.

        Parameters
        ----------
        position : pm.Vec2d
            Initial position of the load body.
        load_mass : float
            Mass of the load. If 0 or negative, mass is calculated from
            shape density. Otherwise, this mass is used directly.

        Returns
        -------
        pm.Body
            The load body with the specified mass.
        """
        load = pm.Body()
        load.position = position
        vertices = (-.5, -.5), (-.5, .5), (.5, .5), (.5, -.5)
        segs: list[pm.Segment] = []
        for i, vertex in enumerate(vertices):
            segment = pm.Segment(
                load, vertex, vertices[(i + 1) % len(vertices)],
                self._thickness
            )
            segment.density = self.density
            # Rigidbodies in this group won't collide
            segment.filter = self.filter
            segs.append(segment)
        self.space.add(load, *segs)

        # If load_mass is specified, override the density-calculated mass
        # This ensures the frame/chassis has the expected mass for stability
        if load_mass > 0:
            load.mass = load_mass
            # Set moment proportional to mass for a square shape
            # moment = m * (w^2 + h^2) / 12 for a rectangle
            load.moment = load_mass * (1.0 + 1.0) / 12.0

        return load


def convert_to_dynamic_linkage(
    kinematic_linkage: Linkage,
    space: pm.Space,
    density: float = 1,
    load: float = 1,
    motor_rate: float | None = None,
) -> DynamicLinkage:
    """Convert a classic Linkage to its dynamic counterpart.

    Parameters
    ----------
    kinematic_linkage : Linkage
        The kinematic linkage to convert.
    space : pm.Space
        The pymunk space.
    density : float, optional
        Density for body mass calculation. Default is 1.
    load : float, optional
        Mass of the load to carry. Default is 1.
    motor_rate : float | None, optional
        Motor angular velocity in rad/s. If None and the linkage has a
        motor_rate attribute (like Walker), that value is used.

    Returns
    -------
    DynamicLinkage
        The dynamic equivalent of the kinematic linkage.
    """
    # Get motor_rate from linkage if it has one (e.g., Walker) and not overridden
    if motor_rate is None and hasattr(kinematic_linkage, 'motor_rate'):
        motor_rate = kinematic_linkage.motor_rate

    return DynamicLinkage(
        kinematic_linkage.joints, space, density=density,
        load=load, name=kinematic_linkage.name, motor_rate=motor_rate
    )
