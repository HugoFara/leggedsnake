#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The dynamiclinkage module is an interface between Pymunk and a kinematic linkage.

It provide various classes to overwrite the pylinkage.Joint objects.
It also includes a DynamicLinkage that inheritates from
pylinkage.linkage.Linkage.
Enventually a handy convert_to_dynamic_linkage methoc can generate a
DynamicLinkage from a pylinkage.linkage.Linkage.
"""
import abc
from math import atan2
import pymunk as pm
from pylinkage import linkage
from pylinkage.geometry import dist, cyl_to_cart


class DynamicJoint(abc.ABC):
    """Dynamic, pymunk compatible equivalent of kinematic Joint."""

    def __init__(self, body0=None, body1=None, space=None, radius=.3,
                 density=1, shape_filter=None):
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
            Simulation space in which the DynamicJoint exist.
            The default is None.
        radius : float, optional
            DynamicJoint will generate hulles of this radius. The default is .3.
        density : float, optional
            Density of the hull, mass will be computed accordingly.
            The default is 1.
        shape_filter : pymunk.shapes.ShapeFilter, optional
            Prevent hulles from colliding with each another. Useful is the same
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

    def __generate_link__(self, body, parent_pos):
        """
        Create a pymunk.Segment between two joints on a body and return it.

        First extremity of the segment is self Joint position, the other is
        parent_pos (a position).

        Parameters
        ----------
        body : pymunk.Body
            The Body you want the link to be added to.
        parent_pos : sequence
            Any sequence of two float. It give th

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
        seg.filter = self.filter
        return seg

    def __generate_body__(self, index=0):
        """
        Create a pymunk.Body and assign it to the specified interface.

        Parameters
        ----------
        index : int, optional
            The interface index to assign the body to.
            0 : assign to self.a
            1 : assign to selb.b
            2 : assign to self.a and self.b
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
                and isinstance(getattr(self, 'joint' + sindex), linkage.Joint)
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
            self.space.add(body, seg)

    def __find_common_body__(self):
        """Find the body linking two joints."""
        joints = self.joint0, self.joint1
        bodies = [set((joints[0]._a,)), set((joints[1]._a,))]
        for body, j in zip(bodies, joints):
            body.add(j._b)
            for sup in j.superposed:
                body.add(sup._a)
                body.add(sup._b)
        inter = bodies[0].intersection(bodies[1])
        if len(inter) == 0:
            message = ('Unable to find a common body between parents joints'
                       ' {} and {}.').format(*joints)
            raise linkage.UnbuildableError(self, message=message)
        if len(inter) == 1:
            return inter.pop()
        else:
            message = ('Multiple common bodies between parents joints'
                       ' {} and {}, parents of {}: {}.')
            raise Exception(message.format(joints[0], joints[1], self, inter))

    def reload(self):
        """
        Reload DynamicJoint coordinates.

        Coordinates only depend of Body position and DO NOT use any linkage
        constraint.

        Returns
        -------
        None.

        """
        self.set_coord(*self._b.local_to_world(self._anchor_b))


class Nail(linkage.Static, DynamicJoint):
    """
    A simple point to follow a rigidbody.

    It is special since it DOES NOT generate bodies.
    """

    def __init__(self, x=0, y=0, name=None, body=None, space=None,
                 radius=.3, density=1, shape_filter=None):
        linkage.Static.__init__(self, x, y, name=name)
        DynamicJoint.__init__(
            self, body0=body, body1=body, space=space, radius=radius,
            density=density, shape_filter=shape_filter)
        self._a = self._b = body
        self.space = space
        self.radius = radius
        self.density = density
        self.filter = shape_filter
        if self._a is not None:
            self.__set_offset__()
            if hasattr(self, 'joint0') and isinstance(self.joint0,
                                                      linkage.Joint):
                seg = pm.Segment(self._a, self._a.world_to_local(self.coord()),
                                 self._a.world_to_local(self.joint0.coord()),
                                 self.radius)
                seg.density = self.density
                seg.filter = self.filter
                self.space.add(seg)
            if hasattr(self, 'joint1') and isinstance(self.joint1,
                                                      linkage.Joint):
                seg = pm.Segment(self._a, self._a.world_to_local(self.coord()),
                                 self._a.world_to_local(self.joint1.coord()),
                                 self.radius)
                seg.density = self.density
                seg.filter = self.filter
                self.space.add(seg)

    def __set_offset__(self):
        """Memorize the offset distance between self and linked Body."""
        self._distance0 = dist(self.coord(), self._a.position)
        x_pos, y_pos = self.coord() - self._a.position
        self._angle0 = atan2(y_pos, x_pos) - self._a.angle

    def reload(self):
        """Reload position based on linked body rotation and position."""
        self.set_coord(cyl_to_cart(self._distance0,
                                   self._a.angle + self._angle0,
                                   self._a.position))


class PinUp(linkage.Fixed, DynamicJoint):
    """
    Dynamic counterpart of Fixed joint.

    Add two pm.Segment to the linked body.
    """

    def __init__(self, x=0, y=0, joint0=None, space=None,
                 joint1=None, distance=None, angle=None, name=None,
                 radius=.3, density=1, shape_filter=None):
        linkage.Fixed.__init__(
            self, x, y, joint0=joint0, joint1=joint1, name=name,
            distance=distance, angle=angle)
        DynamicJoint.__init__(
            self, space=space, radius=radius, density=density,
            shape_filter=shape_filter)
        if (
                isinstance(self.joint0, linkage.Joint) and
                isinstance(self.joint1, linkage.Joint)
        ):
            linkage.Fixed.reload(self)
        if self.joint0 is not None:
            self.set_anchor_a(self.joint0, self.r, self.angle)
        if self.joint1 is not None:
            self.set_anchor_b(self.joint1)

    def set_anchor_a(self, joint, distance=None, angle=None):
        """
        Set first anchor charactertics.

        Parameters
        ----------
        joint : DynamicJoint
            DynamicJoint to use as achor_a.
        distance : float, optional
            Distance to keep constant. The default is None.
        angle : float, optional
            Angle (in radians)  (joint1, joint0, self). The default is None.

        Returns
        -------
        None.

        """
        linkage.Fixed.set_anchor0(self, joint, distance, angle)

    def set_anchor_b(self, joint):
        """
        Set second anchor caracteristics.

        It will create errors if called before anchor_a is properly defined.
        SIDE EFFECT : it create two Segment and add them to self.space.

        Parameters
        ----------
        joint : DynamicJoint
            Joint to use as anchor_b.

        Returns
        -------
        None.

        """
        linkage.Fixed.set_anchor1(self, joint)
        linkage.Fixed.reload(self)
        self._b = self._a = super().__find_common_body__()
        self._anchor_a = self._a.world_to_local(self.coord())
        self.space.add(self.__generate_link__(self._a, joint.coord()))

        self._anchor_b = self._b.world_to_local(self.coord())
        self.space.add(self.__generate_link__(self._a, joint.coord()))

    def reload(self):
        """
        Reload DynamicJoint coordinates.

        Coordinates only depend of Body position and DO NOT use any linkage
        constraint.

        Returns
        -------
        None.

        """
        DynamicJoint.reload(self)


class DynamicPivot(linkage.Pivot, DynamicJoint):
    """Dynamic counterpart of a Pivot joint."""

    def __init__(self, x=0, y=0, joint0=None, space=None,
                 joint1=None, distance0=None, distance1=None, name=None,
                 radius=.3, density=1, shape_filter=None):
        linkage.Pivot.__init__(
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

    def set_anchor_a(self, joint, distance=None):
        """
        Set anchor_a caracteristics.

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
        linkage.Pivot.set_anchor0(self, joint, distance)
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
            self.space.add(pivot)

    def set_anchor_b(self, joint, distance=None):
        """
        Set anchor_b caracteristics.

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
        linkage.Pivot.set_anchor1(self, joint, distance)
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
            self.space.add(self.pivot, pivot_b)

    def reload(self):
        """
        Reload DynamicJoint coordinates.

        Coordinates only depend of Body position and DO NOT use any linkage
        constraint.

        Returns
        -------
        None.

        """
        DynamicJoint.reload(self)


class Motor(linkage.Crank, DynamicJoint):
    """
    A Motor is a crank.

    It makes a link between a body and a sencond link it creates. It attaches
    them with a PivotJoint, and adds a SimpleMotor over it. The bodies are now
    constrained to rotate around one each other.

    The Motor is placed at the extremity of the body it creates.
    """

    def __init__(self, x=None, y=None, joint0=None, space=None,
                 distance=None, angle=None, name=None, radius=.3, density=1,
                 shape_filter=None):
        linkage.Crank.__init__(self, x, y, joint0=joint0,
                               distance=distance, angle=angle, name=name)
        linkage.Crank.reload(self, dt=0)
        DynamicJoint.__init__(self, space=space, density=density,
                              shape_filter=shape_filter, radius=radius)
        if joint0 is not None:
            self.set_anchor_a(joint0, distance=distance)

    def __get_reference_body__(self, body=None):
        """
        Find a body that can be used as reference body.

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
            A valid Body to be used as reference.

        """
        if body is None:
            return self.joint0._a
        if body in (self.joint0._a, self.joint0._b):
            return body
        message = 'Argument body should in {} bodies'
        raise ValueError(message.format(self.joint0))

    def __generate_body__(self):
        """Generate the crank body only."""
        if hasattr(self, 'joint0') and isinstance(self.joint0, linkage.Joint):
            body = pm.Body()
            body.position = (pm.Vec2d(*self.coord()) + self.joint0.coord()) / 2
            seg = self.__generate_link__(body, self.joint0.coord())
            self._a = self._b = body
            self._anchor_b = body.world_to_local(self.coord())
            self.space.add(body, seg)

    def set_anchor_a(self, joint, distance=None):
        """
        Set anchor_a caracteristics.

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
        linkage.Crank.set_anchor0(self, joint, distance)
        ref_body = self.__get_reference_body__()
        if not hasattr(self, '_b'):
            self.__generate_body__()
            self.pivot = pm.PivotJoint(ref_body, self._b, self.joint0.coord())
            self.pivot.collide_bodies = False
            self.actuator = pm.SimpleMotor(self._b, ref_body, self.angle)
            self.joint0.superposed.add(self.pivot)
            self.joint0.superposed.add(self.actuator)
            self.space.add(self.pivot, self.actuator)

    def reload(self):
        """
        Reload DynamicJoint coordinates.

        Coordinates only depend of Body position and DO NOT use any linkage
        constraint.

        Returns
        -------
        None.

        """
        DynamicJoint.reload(self)


class DynamicLinkage(linkage.Linkage):
    """
    Dynamic couterpart of a kinematic linkage.Linkage.

    It has several attributes linked to its dynamic nature and is close to an
    empty shell in the ways you will use it.

    Please not that it carries a load, which is relevant with real-world
    simulation where the weight of legs is small compared to the weight of the
    frame.
    """

    __slots__ = ['joint_to_rigidbodies', 'rigidbodies', 'body', 'height',
                 'mechanical_energy', 'mass', 'space', 'density',
                 '_thickness', 'filter']

    def __init__(self, joints, space, density=1, load=0, name=None,
                 thickness=.1):
        """
        Instanciate a new DynamicLinkage.

        Parameters
        ----------
        joints : linkage.Joint
            Joints to be part of the linkage. Kinematic joints will be
            converted in there dynamic equivalents.
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

        Returns
        -------
        None.

        """
        super().__init__(joints=joints, name=name)
        self.joint_to_rigidbodies = [None] * len(self.joints)
        self.density = density
        self._thickness = thickness
        self.space = space
        self.filter = pm.ShapeFilter(group=1)
        # Add the rigidbody linked to the frame to the first frame's joint
        load_body = self.build_load(self.joints[0].coord(), load)
        load_body.density = load
        self.rigidbodies = [load_body]
        self.body = load_body
        convert = False
        for i, j in enumerate(self.joints):
            if isinstance(j, linkage.Static):
                self.joint_to_rigidbodies[i] = [load_body]
            if isinstance(j, DynamicJoint):
                j.space = self.space
            else:
                convert = True
        if convert:
            self.convert_to_dynamic_joints(self.joints)
        self.mass = sum(map(lambda x: x.mass, self.rigidbodies))
        self._cranks = tuple(j for j in self.joints if isinstance(j, Motor))

    def convert_to_dynamic_joints(self, joints):
        """Convert a kinematic joint in its dynamic counterpart."""
        if not isinstance(self.space, pm.Space):
            raise Exception('Linkage {} Space not defined yet!'.format(self))
        dynajoints = []
        conversion_dict = {}
        common = {
            'space': self.space,
            'radius': self._thickness,
            'density': self.density,
            'shape_filter': self.filter
        }
        for joint in joints:
            common.update({'x': joint.x, 'y': joint.y, 'name': joint.name})
            if isinstance(joint, DynamicJoint):
                djoint = joint
            elif isinstance(joint, linkage.Static):
                djoint = Nail(body=self.body, **common)
            # Joints with at least one reference
            else:
                """
                Useless while qe don't support quick joint definition
                if (
                        isinstance(joint.joint0, linkage.Static)
                        and joint.joint0 not in conversion_dict
                ):
                    conversion_dict[joint.joint0] = joint.joint0
                if (
                        hasattr(joint, "joint1")
                        and isinstance(joint.joint1, linkage.Static)
                        and joint.joint1 not in conversion_dict
                ):
                    conversion_dict[joint.joint1] = joint.joint1
                """
                if isinstance(joint, linkage.Fixed):
                    djoint = PinUp(distance=joint.r, angle=joint.angle,
                                   joint0=conversion_dict[joint.joint0],
                                   joint1=conversion_dict[joint.joint1],
                                   **common)
                elif isinstance(joint, linkage.Crank):
                    djoint = Motor(
                        joint0=conversion_dict[joint.joint0],
                        distance=joint.r, angle=joint.angle,
                        **common
                    )
                elif isinstance(joint, linkage.Pivot):
                    djoint = DynamicPivot(
                        joint0=conversion_dict[joint.joint0],
                        joint1=conversion_dict[joint.joint1],
                        distance0=joint.r0, distance1=joint.r1,
                        **common
                    )
            dynajoints.append(djoint)
            conversion_dict[joint] = djoint
        self.joints = tuple(dynajoints)

    def build_load(self, position, load_mass):
        """Create the load this linkage have to carry."""
        load = pm.Body(load_mass)
        load.position = position
        vertices = (-.5, -.5), (-.5, .5), (.5, .5), (.5, -.5)
        segs = []
        for i, vertex in enumerate(vertices):
            segs.append(pm.Segment(load, vertex,
                                   vertices[(i + 1) % len(vertices)],
                                   self._thickness))
            segs[-1].density = self.density
            # Rigodbodies in this group won't collide
            segs[-1].filter = self.filter
        self.space.add(load, *segs)
        return load


def convert_to_dynamic_linkage(kinematic_linkage, space, density=1,
                               load=1):
    """Convert a classic Linkage to its dynamic counterpart."""
    return DynamicLinkage(kinematic_linkage.joints, space, density=density,
                          load=load, name=kinematic_linkage.name)
