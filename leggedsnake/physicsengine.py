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
from functools import partial
import numpy as np
import pymunk as pm
import pymunk.matplotlib_util
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from pylinkage.geometry import norm, cyl_to_cart, bounding_box
from pylinkage.linkage import Static, Crank, Fixed, Pivot

from . import dynamiclinkage as dlink

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
        # Time between two physics computation
        "physics_period": 0.02,
    },
    # Display parameters
    "camera": {
        # Do you want to follow a system of view whole scene?
        "dynamic_camera": True,
        # Required frames per second
        "fps": 20,
    },
}


def set_space_constraints(space):
    """Set the solver if they are many constraints."""
    constraints = space.constraints
    len_c = len(space.constraints)
    # Number of iterations can be adapted
    space.iterations = int(10 * np.exp(len_c / 60))
    for constraint in constraints:
        if not isinstance(constraint, pm.SimpleMotor) and False:
            constraint.max_force = params["physics"]["max_force"] * (np.exp(
                - len_c / 25) / 2 + .5)
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
        seg = pm.Segment(self.space.static_body, self.road[0], self.road[-1],
                         .1)
        seg.friction = params["ground"]["friction"]
        self.space.add(seg)
        self.linkages = []

    def add_linkage(self, linkage):
        """Add a DynamicLinkage to the simulation."""
        if isinstance(linkage, dlink.DynamicLinkage):
            dynamic_linkage = linkage
        else:
            dynamic_linkage = dlink.convert_to_dynamic_linkage(
                linkage, self.space)
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
                and norm(linkage.body.velocity) < .1):
            linkage_crank.actuator.max_force = params["linkage"]["torque"]
            linkage.height = linkage.body.position.y
            linkage.mechanical_energy = (.5 * linkage.mass
                                         * norm(linkage.body.velocity) ** 2)

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


class VisualWorld(World):
    """Same as parent class World, but with matplotlib objects."""

    def __init__(self, space=None, road_y=-5):
        """
        Instantiate the world and objects to be displayed.

        Parameters
        ----------
        space : pymunk.space.Space, optional
            Space of simulation. The default is None.
        road_y : float, optional
            The ordinate of the ground. Useful when likages have long legs.
            The default is -5.
        """
        super().__init__(space=space, road_y=road_y)
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.linkage_im = []
        # Same for the road
        self.road_im = self.ax.plot([], [], 'k-', animated=False)

    def add_linkage(self, linkage):
        """
        Add a linkage to the world, and create the appropriate Artist objects.

        Parameters
        ----------
        linkage : pylinkage.linkage.Linkage
            The linkage you want to add, can be a Walker.

        Returns
        -------
        None.

        """
        super().add_linkage(linkage)
        # Objects that will allow display
        linkage_im = []
        for j in self.linkages[-1].joints:
            if (
                    isinstance(j, Static)
                    and hasattr(j, 'joint0')
                    and j.joint0 is not None
            ):
                linkage_im.append(self.ax.plot([], [], 'k-', animated=False)[0])
                if hasattr(j, 'joint1') and j.joint1 is not None:
                    linkage_im.append(self.ax.plot([], [], 'k-', animated=False)[0])
            elif isinstance(j, Crank):
                linkage_im.append(self.ax.plot([], [], 'g-', animated=False)[0])
            elif isinstance(j, Fixed):
                linkage_im.append(self.ax.plot([], [], 'r-', animated=False)[0])
                linkage_im.append(self.ax.plot([], [], 'r-', animated=False)[0])
            elif isinstance(j, Pivot):
                linkage_im.append(self.ax.plot([], [], 'b-', animated=False)[0])
                linkage_im.append(self.ax.plot([], [], 'b-', animated=False)[0])
        self.linkage_im.append(linkage_im)

    def draw_linkage(self, linkage_im, joints):
        """Draw the linkage at his current state."""
        a = 0
        for j in joints:
            if hasattr(j, 'joint0') and j.joint0 is not None:
                linkage_im[a].set_data(
                    [j.x, j.joint0.x], [j.y, j.joint0.y]
                )
                a += 1
            if hasattr(j, 'joint1') and j.joint1 is not None:
                linkage_im[a].set_data(
                    [j.x, j.joint1.x], [j.y, j.joint1.y]
                )
                a += 1
        return linkage_im

    def init_visuals(self, colors=None):
        if colors is not None:
            for im, color in zip(self.linkage_im, colors):
                for line in im:
                    if np.isscalar(color):
                        line.set_alpha(color)
                    else:
                        line.set_color(color)

        return self.road_im + [im for im in self.linkage_im]
    
    def reload_visuals(self, opacities=None):
        """Reload the visual components only."""
        center = np.mean([linkage.joints[0].coord() for linkage in self.linkages], axis=0)
        self.fig.suptitle(f"Position: {tuple(map(int, center))}")

        self.road_im[0].set_data(
            [i[0] for i in self.road],
            [i[1] for i in self.road]
        )
        if params["camera"]["dynamic_camera"]:
            self.ax.set_xlim(center[0] - 10, center[0] + 10)
            self.ax.set_ylim(center[1] - 10, center[1] + 10)
        else:
            self.ax.set_ylim(
                min([0] + [min(i.y for i in linkage.joints) for linkage in self.linkages]) - 5,
                max([0] + [max(i.y for i in linkage.joints) for linkage in self.linkages]) + 5
            )

        # Return modified objects for animation optimization
        visual_objects = []
        for linkage, im in zip(self.linkages, self.linkage_im):
            visual_objects += self.draw_linkage(im, linkage.joints)
        visual_objects += self.road_im
        return visual_objects

    def visual_update(self, time=None, opacities=None):
        """
        Update simulation and draw it.
        
        Parameters
        ----------
        time : list | float | None
            When a list, delta-time for physics and display (respectively) 
            Using a float, only delta-time for physics, fps is set with params["camera"]["fps"]
            Setting to None set physics dt to params["simul"]["physics_period"] and fps to params["camera"]["fps"]
        opacities : list of float or None
            Opacity for the drawing of each linkage.
        """
        if time is None:
            dt = params["simul"]["physics_period"]
            fps = params["camera"]["fps"]
        elif isinstance(time, int) or isinstance(time, float):
            dt = time
            fps = params["camera"]["fps"]
        else:
            dt, fps = time
        div = 1 // (dt * fps)
        if div >= 1:
            update_ret = [0, 0]
            for _ in range(int(div)):
                for i, step_update in enumerate(self.update(dt)):
                    update_ret[i] += step_update
            for i, step_update in enumerate(self.update(1 / fps - dt * div)):
                update_ret[i] += step_update
        else:
            update_ret = self.update(dt)
        self.reload_visuals(opacities)
        return update_ret


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
    if isinstance(linkage, dlink.DynamicLinkage):
        data.extend(tuple(i.position) for i in linkage.rigidbodies)
    return bounding_box(data)


def im_debug(world, linkage):
    """Use pymunk debugging for visual debugging."""
    bbox = linkage_bb(linkage)
    world.ax.clear()
    world.ax.set_xlim(int(bbox[3]) - 5, int(bbox[1]) + 5)
    world.ax.set_ylim(int(bbox[2]) - 5, int(bbox[0]) + 5)
    world.ax.scatter(
        [i.x for i in linkage.joints],
        [i.y for i in linkage.joints],
        c='r'
    )
    for j in linkage.joints:
        for shape in j._a.shapes:
            begin = j._a.local_to_world(shape.a)
            end = j._a.local_to_world(shape.b)
            world.ax.plot([begin[0], end[0]], [begin[1], end[1]])
    options = pymunk.matplotlib_util.DrawOptions(world.ax)
    options.constraint_color = (.1, .1, .1, .0)
    world.space.debug_draw(options)


def video_debug(linkage):
    """Launch the simulation frame by frame, useful for debug."""
    road_y = linkage_bb(linkage)[0] - 1
    if isinstance(linkage, dlink.DynamicLinkage):
        world = VisualWorld(linkage.space, road_y=road_y)
    else:
        world = VisualWorld(road_y=road_y)
    world.add_linkage(linkage)
    dynamic_linkage = world.linkages[-1]
    for _ in range(1, int(1e3)):
        dt = params["simul"]["physics_period"]
        world.space.step(dt)
        recalc_linkage(dynamic_linkage)
        im_debug(world, dynamic_linkage)
        plt.pause(.2)


def all_linkages_video(linkages, duration=30, save=False, colors=None):
    """
    Give the rigidbody a dynamic model and launch simulation with video.

    Parameters
    ----------
    linkages : Union[
        list of pylinkage.linkage.Linkage,
        list of leggedsnake.dynamiclinkage.DynamicLinkage
    ]
        The Linkage you want to simulate.
    duration : float, optional
        Duration (in seconds) of the simulation. The default is 40.
    save : bool, optional
        If you want to save it as a .mp4 file.
    """
    road_y = min(linkage_bb(linkage)[0] for linkage in linkages) - 1
    if isinstance(linkages[0], dlink.DynamicLinkage):
        world = VisualWorld(linkages[0].space, road_y=road_y)
    else:
        world = VisualWorld(road_y=road_y)
    for linkage in linkages:
        world.add_linkage(linkage)
    # Number of frames for the selected duration
    n_frames = int(params["camera"]["fps"] * duration)

    dt = params["simul"]["physics_period"]
    fps = params["camera"]["fps"]
    if dt * fps > 1:
        print(
            f"Warning: Physics is computed every {dt}s ({1 / dt} times/s)",
            f"but display is {fps} times/s."
        )

    if colors is None:
        colors = np.logspace(0, -1, num=len(linkages))
    animation = anim.FuncAnimation(
        world.fig, world.visual_update,
        frames=[None] * (n_frames - 1),
        init_func=partial(world.init_visuals, colors),
        interval=int(1000 / params["camera"]["fps"]),
        repeat=False, blit=False
    )
    if save:
        writer = anim.FFMpegWriter(fps=params["camera"]["fps"], bitrate=2500)
        animation.save(f"Dynamic {linkage[0].name}.mp4", writer=writer)
    else:
        plt.show()
        if animation:
            pass


def video(linkage, duration=30, save=False):
    """
    Give the rigidbody a dynamic model and launch simulation with video.

    Parameters
    ----------
    linkage : Union[pylinkage.linkage.Linkage,
    leggedsnake.dynamiclinkage.DynamicLinkage]
        The Linkage you want to simulate.
    duration : float, optional
        Duration (in seconds) of the simulation. The default is 40.
    save : bool, optional
        If you want to save it as a .mp4 file.
    """
    all_linkages_video([linkage], duration, save)


if __name__ == "__main__":
    base = Static(0, 0, name="Main trick")
    crank = Crank(1, 0, name="The crank", angle=1, joint0=base)
    follower = Pivot(0, 2, joint0=base, joint1=crank, distance0=2,
                     distance1=1)
    frame = Fixed(joint0=crank, joint1=follower, distance=1, angle=-np.pi/2)
    demo_linkage = dlink.DynamicLinkage(
        name='Some tricky linkage',
        joints=(base, crank, follower, frame),
        space=pm.Space()
    )
    demo_linkage.space.gravity = params["physics"]["gravity"]
    video_debug(demo_linkage)
