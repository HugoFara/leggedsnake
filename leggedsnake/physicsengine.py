# -*- coding: utf-8 -*-
"""
The physicsengine module gives a dynamic behavior to legged mechanism.

It uses the 2D physics engine chipmunk, this is why it can only be used on
planar mechanisms.
In theory, you can use any type of mechanism, and not only planar mechanisms.
In practice, we do generate the road and some other parameters as the gravity,
so it can be difficult to test something other than a walker.

Created on Sat May 25 2019 14:56:01.

@author: HugoFara
"""
import numpy as np
import pymunk as pm
from pylinkage.geometry import norm, cyl_to_cart, bounding_box
from pylinkage.linkage import Static, Crank, Fixed, Pivot

from . import dynamiclinkage

# Simulation parameters
params = {
    # Ground parameters
    "ground": {
        # Nominal slope (radian)
        "slope": 10 * np.pi / 180,
        # Maximal step height
        "max_step": .5,
        # Steps frequency
        "step_freq": .1,
        # Terrain variations should not be above 1
        "noise": .9,
        # Road trunks length
        "section_len": 1,
        # Ground friction coefficient root
        "friction": .5 ** .5
    },
    # Studied system parameters
    "linkage": {
        # Maximal torque (N.m)
        "torque": 1e3,
        # Crank length (m) (unused for now)
        "crank_len": .05,
        # Linear mass of bars (kg/m)
        "masses": 1,
        # Load mass (kg)
        "load": 10,
    },
    # Physics engine parameters
    "physics": {
        "gravity": (0, -9.80665),
        # Maximal value of forces (N)
        "max_force": 1e10,
    },
    # Study hypothesis
    "simul": {
        # Time between two physics computations
        "physics_period": 0.02,
    }
}


def set_space_constraints(space):
    """Set the solver if they are many constraints."""
    constraints = space.constraints
    len_c = len(space.constraints)
    # Number of iterations can be adapted
    space.iterations = int(10 * np.exp(len_c / 60))
    for constraint in constraints:
        if not isinstance(constraint, pm.SimpleMotor) and False:
            constraint.max_force = params["physics"]["max_force"] * (
                    np.exp(- len_c / 25) / 2 + .5
            )
        constraint.error_bias = (1 - .1 * np.exp(-len_c / 60)) ** 60


class World:
    """
    A world contains a space of simulation, at least one linkage, and a road.

    It is not intended to be rendered visually per se, see VisualWorld for
    this purpose.
    """

    def __init__(self, space=None, road_y=-5):
        """
        Initiate rigidbodies and simulation.

        Add rigidbodies in linkage.

        Parameters
        ----------
        space : pymunk.space.Space, optional
            Space of simulation. The default is None.
        road_y : float, optional
            The ordinate of the ground. Useful when linkages have long legs.
            The default is -5.
        """
        if isinstance(space, pm.Space):
            self.space = space
        else:
            self.space = pm.Space()
            self.space.gravity = params["physics"]["gravity"]

        set_space_constraints(self.space)

        # The road which will be built
        self.road = [(-15, road_y), (15, road_y)]
        # First road parts
        seg = pm.Segment(
            self.space.static_body, self.road[0], self.road[-1],
            .1
        )
        seg.friction = params["ground"]["friction"]
        self.space.add(seg)
        self.linkages = []

    def add_linkage(self, linkage):
        """Add a DynamicLinkage to the simulation."""
        if isinstance(linkage, dynamiclinkage.DynamicLinkage):
            dynamic_linkage = linkage
        else:
            dynamic_linkage = dynamiclinkage.convert_to_dynamic_linkage(
                linkage, self.space
            )
        for cur_crank in dynamic_linkage._cranks:
            cur_crank.actuator.max_force = 0
        self.linkages.append(dynamic_linkage)
        for s in self.space.shapes:
            s.friction = params["ground"]["friction"]
        # set_space_constraints(self.space)

    def __update_linkage__(self, linkage, power):
        """Update a specific linkage."""
        linkage_crank = next(j for j in linkage.joints if isinstance(j, Crank))
        if (
                linkage_crank.actuator.max_force == 0
                and norm(linkage.body.velocity) < .1
        ):
            linkage_crank.actuator.max_force = params["linkage"]["torque"]
            linkage.height = linkage.body.position.y
            linkage.mechanical_energy = (
                    .5 * linkage.mass * norm(linkage.body.velocity) ** 2
            )

        # Energy from the motor in this step
        energy = power * params["simul"]["physics_period"]
        if hasattr(linkage, 'height') and energy != 0.:
            v = norm(linkage.body.velocity)
            g = norm(params["physics"]["gravity"])
            m = linkage.mass
            new_mechanical_energy = m * (
                .5 * v ** 2 + g * (linkage.body.position.y - linkage.height)
            )
            efficiency = (
                new_mechanical_energy - linkage.mechanical_energy
            ) / energy
            linkage.mechanical_energy = new_mechanical_energy
            return energy, efficiency
        return 0, 0

    def update(self, dt=None):
        """
        Update simulation.
        
        Parameters
        ----------
        dt : float | None
            Time of the step (delta-time). Uses params["simul"]["physics_period"] if None.
        """
        # Simulation step
        if dt is None:
            dt = params["simul"]["physics_period"]
        # Motor power in this simulation step
        powers = [
            [0 for j in lin.joints if isinstance(j, Crank)] 
            for lin in self.linkages
        ]
        self.space.step(dt)
        for i, linkage in enumerate(self.linkages):
            index = -1
            for crank in linkage.joints:
                if not isinstance(crank, Crank):
                    continue
                index += 1
                # Get offset for crank rotation speed
                w = crank._b.angular_velocity
                w -= linkage.body.angular_velocity
                powers[i][index] += abs(w) * crank.actuator.impulse / dt

        bounds = (0, 0)
        energies = [0] * len(self.linkages)
        efficiencies = [0] * len(self.linkages)
        for i, linkage, power in zip(
                range(len(self.linkages)), self.linkages, powers
        ):
            recalc_linkage(linkage)
            energies[i], efficiencies[i] = self.__update_linkage__(
                linkage, power[0]
            )
            bounds = (
                min(bounds[0], *(i.x for i in linkage.joints)),
                max(bounds[1], *(i.x for i in linkage.joints))
            )
        while self.road[-1][0] < bounds[1] + 10:
            self.build_road(True)
        while self.road[0][0] > bounds[0] - 10:
            self.build_road(False)

        # Without animation, we return 100 times motor yield
        # with a duration step
        for linkage, energy, efficiency in zip(
                self.linkages, energies, efficiencies
        ):
            return efficiency, energy * dt

    def __build_road_step__(self, ground, index):
        """Add a step (two points)."""
        high = np.random.rand() * ground["max_step"]
        a = self.road[index][0], self.road[index][1] + high
        b = (
            self.road[index][0] + ground["section_len"] * (1 - index),
            self.road[index][1] + high
        )

        s = pm.Segment(self.space.static_body, a, b, .1)
        s.friction = ground["friction"]
        self.space.add(s)
        s = pm.Segment(self.space.static_body, a, self.road[index], .1)
        s.friction = ground["friction"]
        self.space.add(s)
        # Add the elements in the end or the beginning
        self.road.insert(-index * len(self.road), a)
        self.road.insert(-index * len(self.road), b)

    def __build_road_segment__(self, ground, index):
        """Add a segment (one point)."""
        # Add noise for more chaotic terrain."""
        angle = np.random.normal(
            ground["slope"] / 2, ground["noise"] * ground["slope"] / 2
        )
        # Adding a point to the left increases angle by pi/2
        if not index:
            angle = np.pi - angle
        a = pm.Vec2d(*cyl_to_cart(ground["section_len"], angle,
                                  self.road[index]))
        s = pm.Segment(self.space.static_body, a, self.road[index], .1)
        s.friction = ground["friction"]
        self.space.add(s)
        self.road.insert(-index * len(self.road), a)

    def build_road(self, positive=False):
        """
        Build a road part.

        Arguments
        ---------
        positive: if False (default), the road part will be added on the left.
        """
        # Ground parameters
        ground = params["ground"]
        if np.random.rand() < ground["step_freq"] and False:
            self.__build_road_step__(ground, -positive)
        else:
            self.__build_road_segment__(ground, -positive)


def recalc_linkage(linkage):
    """Assign a good position to all joints."""
    for j in linkage.joints:
        j.reload()


def linkage_bb(linkage):
    """
    Return the bounding box for this linkage.

    The bounding box is in form (min_y, max_x, max_y, min_x).

    Parameters
    ----------
    linkage : pylinkage.linkage.Linkage
        The linkage from which to get the bounding box.
    """
    data = [i.coord() for i in linkage.joints]
    if isinstance(linkage, dynamiclinkage.DynamicLinkage):
        data.extend(tuple(i.position) for i in linkage.rigidbodies)
    return bounding_box(data)


if __name__ == "__main__":
    base = Static(0, 0, name="Main trick")
    crank = Crank(1, 0, name="The crank", angle=1, joint0=base)
    follower = Pivot(0, 2, joint0=base, joint1=crank, distance0=2,
                     distance1=1)
    frame = Fixed(joint0=crank, joint1=follower, distance=1, angle=-np.pi/2)
    demo_linkage = dynamiclinkage.DynamicLinkage(
        name='Some tricky linkage',
        joints=(base, crank, follower, frame),
        space=pm.Space()
    )
    demo_linkage.space.gravity = params["physics"]["gravity"]
