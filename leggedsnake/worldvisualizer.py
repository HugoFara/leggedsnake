"""
This file contains class and method to see the walkers in action.
"""
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pymunk.matplotlib_util
import matplotlib.animation as anim
from pymunk import Space

from . import physicsengine as pe
from . import dynamiclinkage

# Display settings
CAMERA_SETTINGS = {
    # Do you want to follow a system of view whole scene?
    "dynamic_camera": False,
    # Required frames per second
    "fps": 20,
}


class VisualWorld(pe.World):
    """Same as parent class World, but with matplotlib objects."""

    def __init__(self, space=None, road_y=-5):
        """
        Instantiate the world and objects to be displayed.

        Parameters
        ----------
        space : pymunk.space.Space, optional
            Space of simulation. The default is None.
        road_y : float, optional
            The ordinate of the ground. Useful when linkages have long legs.
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
                    isinstance(j, pe.Static)
                    and hasattr(j, 'joint0')
                    and j.joint0 is not None
            ):
                linkage_im.append(self.ax.plot([], [], 'k-', animated=False)[0])
                if hasattr(j, 'joint1') and j.joint1 is not None:
                    linkage_im.append(self.ax.plot([], [], 'k-', animated=False)[0])
            elif isinstance(j, pe.Crank):
                linkage_im.append(self.ax.plot([], [], 'g-', animated=False)[0])
            elif isinstance(j, pe.Fixed):
                linkage_im.append(self.ax.plot([], [], 'r-', animated=False)[0])
                linkage_im.append(self.ax.plot([], [], 'r-', animated=False)[0])
            elif isinstance(j, pe.Pivot):
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

    def reload_visuals(self):
        """Reload the visual components only."""
        center = np.mean([linkage.joints[0].coord() for linkage in self.linkages], axis=0)
        self.fig.suptitle(f"Position: {tuple(map(int, center))}")

        self.road_im[0].set_data(
            [i[0] for i in self.road],
            [i[1] for i in self.road]
        )
        if CAMERA_SETTINGS["dynamic_camera"]:
            self.ax.set_xlim(center[0] - 10, center[0] + 10)
            self.ax.set_ylim(center[1] - 10, center[1] + 10)
        else:
            self.ax.set_xlim(
                min([0] + [min(i.x for i in linkage.joints) for linkage in self.linkages]) - 10,
                max([0] + [max(i.x for i in linkage.joints) for linkage in self.linkages]) + 10
            )
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

    def visual_update(self, time=None):
        """
        Update simulation and draw it.

        Parameters
        ----------
        time : list | float | None
            When a list, delta-time for physics and display (respectively)
            Using a float, only delta-time for physics, fps is set with CAMERA_SETTINGS["fps"]
            Setting to None set physics dt to pe.params["simul"]["physics_period"] and fps to CAMERA_SETTINGS["fps"]
        """
        if time is None:
            dt = pe.params["simul"]["physics_period"]
            fps = CAMERA_SETTINGS["fps"]
        elif isinstance(time, int) or isinstance(time, float):
            dt = time
            fps = CAMERA_SETTINGS["fps"]
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
        self.reload_visuals()
        return update_ret


def im_debug(world, linkage):
    """Use pymunk debugging for visual debugging."""
    bbox = pe.linkage_bb(linkage)
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
    road_y = pe.linkage_bb(linkage)[0] - 1
    if isinstance(linkage, dynamiclinkage.DynamicLinkage):
        world = VisualWorld(linkage.space, road_y=road_y)
    else:
        world = VisualWorld(road_y=road_y)
    world.add_linkage(linkage)
    dynamic_linkage = world.linkages[-1]
    for _ in range(1, int(1e3)):
        dt = pe.params["simul"]["physics_period"]
        world.space.step(dt)
        pe.recalc_linkage(dynamic_linkage)
        im_debug(world, dynamic_linkage)
        plt.pause(.2)


def all_linkages_video(linkages, duration=30, save=False, colors=None, dynamic_camera=False):
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
    colors : list of float or list of list of float
        * If a list of float, it is the list of opacities
        * If a list of list of float, it is the list of colors
        * If None, opacities are set randomly
    dynamic_camera : bool, optional
        Type of visualization. True follows one strider, False gives a larger view.
        The default is False.
    """
    road_y = min(pe.linkage_bb(linkage)[0] for linkage in linkages) - 1
    if isinstance(linkages[0], dynamiclinkage.DynamicLinkage):
        world = VisualWorld(linkages[0].space, road_y=road_y)
    else:
        world = VisualWorld(road_y=road_y)
    for linkage in linkages:
        world.add_linkage(linkage)
    # Number of frames for the selected duration
    n_frames = int(CAMERA_SETTINGS["fps"] * duration)

    dt = pe.params["simul"]["physics_period"]
    fps = CAMERA_SETTINGS["fps"]
    if dt * fps > 1:
        print(
            f"Warning: Physics is computed every {dt}s ({1 / dt} times/s)",
            f"but display is {fps} times/s."
        )

    if colors is None:
        colors = np.logspace(0, -1, num=len(linkages))
    previous_camera = CAMERA_SETTINGS["dynamic_camera"]
    CAMERA_SETTINGS["dynamic_camera"] = dynamic_camera
    animation = anim.FuncAnimation(
        world.fig, world.visual_update,
        frames=[None] * (n_frames - 1),
        init_func=partial(world.init_visuals, colors),
        interval=int(1000 / CAMERA_SETTINGS["fps"]),
        repeat=False, blit=False
    )
    if save:
        writer = anim.FFMpegWriter(fps=CAMERA_SETTINGS["fps"], bitrate=2500)
        animation.save(f"Dynamic {linkages[0].name}.mp4", writer=writer)
    else:
        plt.show()
        if animation:
            pass
    CAMERA_SETTINGS["dynamic_camera"] = previous_camera


def video(linkage, duration=30, save=False, dynamic_camera=True):
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
    dynamic_camera : bool, optional
        Type of visualization. True follows one strider, False gives a larger view.
        The default is True.
    """
    all_linkages_video([linkage], duration, save, dynamic_camera=dynamic_camera)


if __name__ == "__main__":
    base = pe.Static(0, 0, name="Main trick")
    crank = pe.Crank(1, 0, name="The crank", angle=1, joint0=base)
    follower = pe.Pivot(
        0, 2, joint0=base, joint1=crank, distance0=2, distance1=1
    )
    frame = pe.Fixed(joint0=crank, joint1=follower, distance=1, angle=-np.pi/2)
    demo_linkage = pe.dynamiclinkage.DynamicLinkage(
        name='Some tricky linkage',
        joints=(base, crank, follower, frame),
        space=Space()
    )
    demo_linkage.space.gravity = pe.params["physics"]["gravity"]
    video_debug(demo_linkage)
