#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 18:49:07 2021.

@author: HugoFara

Interface between pymunk and a kinematic linkage
"""
from abc import ABC
from math import atan2
import pymunk as pm
from pylinkage import linkage
from pylinkage.geometry import dist, cyl_to_cart


class DynamicJoint(ABC):
    """Dynamic, pymunk compatible equivalent of kinematic Joint."""

    #__slots__ = ['body0', 'bpdy1', '_distance0', '_distance1', '_angle0',
    #             '_angle1', 'space']

    def __init__(self, body0=None, body1=None, space=None, radius=.3,
                 density=1, shape_filter=None):
        #super(linkage.Joint, self).__init__(x, y, name=name)
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
        seg = pm.Segment(
            body, body.world_to_local(self.coord()),
            body.world_to_local(parent_pos),
            self.radius)
        seg.density = self.density
        seg.filter = self.filter
        return seg

    def __generate_body__(self, index=0):
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
            #self.__set_offset__(1)

    def reload(self):
        self.set_coord(*self.b.local_to_world(self._anchor_b))

class Nail(linkage.Static, DynamicJoint):
    """
    A simple point to follow a rigidbody.

    It is special since it DOES NOT generate rigidbodies.
    """
    def __init__(self, x=0, y=0, name=None, body=None, space=None,
                 radius=.3, density=1, shape_filter=None):
        super(linkage.Static, self).__init__(x, y, name=name)
        DynamicJoint.__init__(
            self, body0=body, body1=body, space=space, radius=radius,
            density=density, shape_filter=shape_filter)
        self.a = self.b = body
        self.space = space
        self.radius = radius
        self.density = density
        self.filter = shape_filter
        if self.a is not None:
            self.__set_offset__()
            if hasattr(self, 'joint0') and isinstance(self.joint0,
                                                      linkage.Joint):
                seg = pm.Segment(self.a, self.a.world_to_local(self.coord()),
                                 self.a.world_to_local(self.joint0.coord()),
                                 self.radius)
                seg.density = self.density
                seg.filter = self.filter
                self.space.add(seg)
            if hasattr(self, 'joint1') and isinstance(self.joint1,
                                                      linkage.Joint):
                seg = pm.Segment(self.a, self.a.world_to_local(self.coord()),
                                 self.a.world_to_local(self.joint1.coord()),
                                 self.radius)
                seg.density = self.density
                seg.filter = self.filter
                self.space.add(seg)

    def __set_offset__(self):
        self._distance0 = dist(self.coord(), self.a.position)
        x, y = self.coord() - self.a.position
        self._angle0 = atan2(y, x) - self.a.angle

    def reload(self):
        self.set_coord(cyl_to_cart(self._distance0,
                                   self.a.angle + self._angle0,
                                   self.a.position))

class PinUp(linkage.Fixed, DynamicJoint):
    """Dynamic counterpart of Fixed joint. Will add segments to the linked
    body"""

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
            self.set_anchor0(self.joint0, self.r, self.angle)
        if self.joint1 is not None:
            self.set_anchor1(self.joint1)

    def set_anchor0(self, joint, distance=None, angle=None):
        linkage.Fixed.set_anchor0(self, joint, distance, angle)

    def set_anchor1(self, joint):
        linkage.Fixed.set_anchor1(self, joint)
        linkage.Fixed.reload(self)
        self.b = self.a = self.__find_common_body__()
        self.anchor_a = self.a.world_to_local(self.coord())
        self._anchor_a = self.anchor_a
        self.space.add(self.__generate_link__(self.a, joint.coord()))

        self.anchor_b = self.b.world_to_local(self.coord())
        self._anchor_b = self.anchor_b
        self.space.add(self.__generate_link__(self.a, joint.coord()))

    def __find_common_body__(self):
        """Find the body linking two joints."""
        joints = self.joint0, self.joint1
        bodies = [set((joints[0].a,)), set((joints[1].a,))]
        for body, j in zip(bodies, joints):
            body.add(j.b)
            #if not isinstance(j, (PinUp, Nail)):
            #    continue
            for sup in j.superposed:
                body.add(sup.a)
                body.add(sup.b)
        inter = bodies[0].intersection(bodies[1])
        if len(inter) == 0:
            message = ('Unable to find a common body between parents joints'
                       ' {} and {}.').format(*joints)
            raise linkage.UnbuildableError(self, message=message)
        elif len(inter) == 1:
            return inter.pop()
        else:
            message = ('Multiple common bodies between parents joints'
                       ' {} and {}, parents of {}: {}.')
            raise Exception(message.format(joints[0], joints[1], self, inter))

    def reload(self):
        DynamicJoint.reload(self)

class DynamicPivot(linkage.Pivot, DynamicJoint, pm.constraints.PivotJoint):
    """Dynamic counterpart. """

    def __init__(self, x=0, y=0, joint0=None, space=None,
                 joint1=None, distance0=None, distance1=None, name=None,
                 radius=.3, density=1, shape_filter=None):
        linkage.Pivot.__init__(
            self, x, y, joint0=joint0, joint1=joint1, name=name,
            distance0=distance0, distance1=distance1)
        DynamicJoint.__init__(
            self, space=space, radius=radius, density=density,
            shape_filter=shape_filter)

        if self.joint0 is not None:
            self.set_anchor0(self.joint0, self.r0)
        if self.joint1 is not None:
            self.set_anchor1(self.joint1, self.r1)

    def set_anchor0(self, joint, distance=None):
        linkage.Pivot.set_anchor0(self, joint, distance)
        if not hasattr(self, 'a'):
            self.__generate_body__(0)
            if isinstance(joint, Nail):
                parent_body0 = joint.a
            else:
                parent_body0 = joint.b
            pivot = pm.PivotJoint(
                self.a, parent_body0,
                joint.coord())
            pivot.collide_bodies = False
            joint.superposed.add(pivot)
            self.space.add(pivot)

    def set_anchor1(self, joint, distance=None):
        linkage.Pivot.set_anchor1(self, joint, distance)
        if not hasattr(self, 'b'):
            self.__generate_body__(1)
            pm.PivotJoint.__init__(
                self, self._a, self._b,
                self.coord())
            self.collide_bodies = False
            if isinstance(joint, Nail):
                parent_body1 = joint.a
            else:
                parent_body1 = joint.b
            pivot = pm.PivotJoint(
                self.b, parent_body1,
                joint.coord()
                )
            pivot.collide_bodies = False
            joint.superposed.add(pivot)
            self.space.add(self, pivot)


    def reload(self):
        DynamicJoint.reload(self)

class Motor(linkage.Crank, DynamicJoint):
    """
    A Motor is a crank.

    It makes a link between a body and a sencond link it creates. It attaches
    them with a PivotJoint, and adds a SimpleMotor over it. The bodies are now
    constrained to rotate around one each other.

    The Motor is placed at the extremity of the body it creates.
    """

    def __init__(self, x=None, y=None, joint0=None, joint1=None, space=None,
                 distance=None, angle=None, name=None, radius=.3, density=1,
                 shape_filter=None):
        linkage.Crank.__init__(self, x, y, joint0=joint0,
                               distance=distance, angle=angle, name=name)
        linkage.Crank.reload(self, dt=0)
        DynamicJoint.__init__(self, space=space, density=density,
                              shape_filter=shape_filter)
        if joint0 is not None:
            self.set_anchor0(joint0, distance=distance)

    def __set_ref_body__(self, body=None):
        if body is None:
            return self.joint0.a
        elif body in (self.joint0.a, self.joint0.b):
            return body
        else:
            message = 'Argument body should in {} bodies'
            raise ValueError(message.format(self.joint0))

    def __generate_body__(self):
        """ Generate the crank body only."""
        if hasattr(self, 'joint0') and isinstance(self.joint0, linkage.Joint):
            body = pm.Body()
            body.position = (pm.Vec2d(*self.coord()) + self.joint0.coord())/2
            seg = self.__generate_link__(body, self.joint0.coord())
            self.a = self.b = body
            self._anchor_b = body.world_to_local(self.coord())
            self.space.add(body, seg)

    def set_anchor0(self, joint, distance=None):
        linkage.Crank.set_anchor0(self, joint, distance)
        ref_body = self.__set_ref_body__()
        if not hasattr(self, 'b'):
            self.__generate_body__()
            self.pivot = pm.PivotJoint(ref_body, self.b, self.joint0.coord())
            self.pivot.collide_bodies = False
            self.actuator = pm.SimpleMotor(self.b, ref_body, self.angle)
            self.joint0.superposed.add(self.pivot)
            self.joint0.superposed.add(self.actuator)
            self.space.add(self.pivot, self.actuator)

    def reload(self):
        DynamicJoint.reload(self)


class DynamicLinkage(linkage.Linkage):

    __slots__ = ['joint_to_rigidbodies', 'rigidbodies', 'body', 'height',
                 'mechanical_energy', 'mass', 'space', 'density',
                 '_thickness', 'filter']

    def __init__(self, joints, space, density=1, load=0, name=None,
                 thickness=.1):
        """
        Instanciate a new DynamicLinkage.

        Arguments
        ---------
        joints: joints to be part of the linkage. Kinematic joints will
        be converted in there dynamic equivalent.
        space: pymunk Space in which linkage should be instantiated
        density: density of the Bodies in the linkage
        load: mass of the load (default: 0)
        name: user-friendly name for the linkage.
        thickness: ratio bar length/width for each bar (radius in pymunk)
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
        common = {'space': self.space, 'radius': self._thickness,
                  'density': self.density, 'shape_filter': self.filter}
        for joint in joints:
            common.update({'x': joint.x, 'y': joint.y, 'name': joint.name})
            if isinstance(joint, DynamicJoint):
                djoint = joint
            elif isinstance(joint, linkage.Static):
                djoint = Nail(body=self.body, **common)
            elif isinstance(joint, linkage.Fixed):
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
        """The load this linkage should have."""
        load = pm.Body(load_mass)
        load.position = position
        #load.angle = pi / 2
        vertices = (-.5, -.5), (-.5, .5), (.5, .5), (.5, -.5)
        segs = []
        for i, v in enumerate(vertices):
            segs.append(pm.Segment(load, v, vertices[(i+1) % len(vertices)],
                                  self._thickness))
            segs[-1].density = self.density
            # Rigodbodies in this group won't collide
            segs[-1].filter = self.filter
        self.space.add(load, *segs)
        return load

    def convert_to_dynamic_linkage(linkage, space):
        """Convert a classic Linkage to its dynamic counterpart."""
        return DynamicLinkage(linkage.joints, space, density=1, load=1,
                              name=linkage.name)