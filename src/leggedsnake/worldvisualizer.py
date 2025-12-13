"""
This file contains class and method to see the walkers in action.
"""
from __future__ import annotations

from functools import partial
from typing import Any, TypedDict

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import numpy.typing as npt
import pymunk as pm
import pymunk.matplotlib_util
import matplotlib.animation as animation
from pymunk import Space
from pylinkage import Static, Crank, Fixed, Pivot
from pylinkage.linkage import Linkage

from . import physicsengine as pe
from . import dynamiclinkage


class CameraSettings(TypedDict):
    dynamic_camera: bool
    fps: int

# Display settings
CAMERA: CameraSettings = {
    # Do you want to follow a system of view whole scene?
    "dynamic_camera": False,
    # Required frames per second
    "fps": 20,
}

# Type alias for bounds
Bounds = tuple[tuple[float, float], tuple[float, float]]


def smooth_transition(
    target: Bounds,
    prev_view: Bounds,
    dampers: Bounds = ((-10, -5), (10, 5)),
) -> list[list[float]]:
    """
    Create a smooth transition between a camera view (prev_view) and a target view.

    Parameters
    ----------
    target : tuple of tuple of float
        Target camera bounds.
    prev_view : tuple of tuple of float
        Current camera bounds
    dampers : tuple of tuple of float
        Absolute values to stay in

    Returns
    -------
        New bounds : tuple of tuple of float
    """
    new_bounds: list[list[float]] = [list(target[0]), list(target[1])]
    # Below this reactivity, we won't resize the window. Does not seem to work.
    reactivity_threshold = 0.5
    reactivity: list[list[float]] = [[0.0, 0.0], [0.0, 0.0]]
    for i in range(2):
        for j in range(2):
            operator = max if j else min
            if operator(target[i][j], prev_view[i][j]) == prev_view[i][j]:
                # We are in-bounds
                # do not change anything
                if operator(target[i][j] - dampers[i][j], prev_view[i][j]) == prev_view[i][j]:
                    reactivity[i][j] = float(np.interp(
                        target[i][j],
                        (prev_view[i][j], prev_view[i][j] + dampers[i][j]),
                        [0, 1]
                    ))
                else:
                    reactivity[i][j] = 0.0
            elif operator(target[i][j] + dampers[i][j], prev_view[i][j]) == prev_view[i][j]:
                # Damper zone, initiate a smooth transition
                reactivity[i][j] = float(np.interp(
                    target[i][j],
                    (prev_view[i][j], prev_view[i][j] - dampers[i][j]),
                    [0, 1]
                ))
            else:
                # Out-of-bounds, move a quick as possible
                reactivity[i][j] = 1.0
    if any(reac[0] > reactivity_threshold or reac[1] > reactivity_threshold for reac in reactivity):
        for i in range(2):
            for j in range(2):
                new_bounds[i][j] = target[i][j] * reactivity[i][j] + prev_view[i][j] * (1 - reactivity[i][j])
    return new_bounds


class VisualWorld(pe.World):
    """Same as parent class World, but with matplotlib objects."""

    fig: Figure
    ax: Axes
    linkage_im: list[list[Line2D]]
    road_im: list[Line2D]

    def __init__(self, space: pm.Space | None = None, road_y: float = -5) -> None:
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

    def add_linkage(
        self, linkage: Linkage | dynamiclinkage.DynamicLinkage, load: float = 0
    ) -> None:
        """
        Add a linkage to the simulation, and create the appropriate Artist objects.

        Parameters
        ----------
        linkage : pylinkage.Linkage or leggedsnake.DynamicLinkage
            Linkage to add.
            We use the linkage's space if it is a DynamicLinkage.
        load : float, optional
            Load to add the center of the linkage.
            Has no effect when using a DynamicLinkage.
            The default is 0.
        Returns
        -------
        None.

        """
        super().add_linkage(linkage, load)
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

    def draw_linkage(
        self, linkage_im: list[Line2D], joints: tuple[Any, ...]
    ) -> list[Line2D]:
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

    def init_visuals(
        self, colors: list[float] | list[list[float]] | npt.NDArray[np.floating[Any]] | None = None
    ) -> list[Any]:
        if colors is not None:
            for im, color in zip(self.linkage_im, colors):
                for line in im:
                    if isinstance(color, (int, float, np.floating)):
                        line.set_alpha(float(color))
                    else:
                        line.set_color(color)

        return self.road_im + [im for im in self.linkage_im]

    def reload_visuals(self) -> list[Line2D]:
        """Reload the visual components only."""
        center = np.mean([linkage.joints[0].coord() for linkage in self.linkages], axis=0)
        self.fig.suptitle(f"Position: {tuple(map(int, center))}")

        self.road_im[0].set_data(
            [i[0] for i in self.road],
            [i[1] for i in self.road]
        )
        prev_view: Bounds = self.ax.get_xlim(), self.ax.get_ylim()
        target: Bounds
        if CAMERA["dynamic_camera"]:
            target = (float(center[0]) - 10, float(center[0]) + 10), (float(center[1]) - 10, float(center[1]) + 10)
        else:
            target = (
                min([0.0] + [min(float(i.x) for i in linkage.joints) for linkage in self.linkages]) - 10,
                max([0.0] + [max(float(i.x) for i in linkage.joints) for linkage in self.linkages]) + 10
            ), (
                min([0.0] + [min(float(i.y) for i in linkage.joints) for linkage in self.linkages]) - 5,
                max([0.0] + [max(float(i.y) for i in linkage.joints) for linkage in self.linkages]) + 5
            )
        new_bounds = smooth_transition(target, prev_view)
        self.ax.set_xlim(*new_bounds[0])
        self.ax.set_ylim(*new_bounds[1])
        # Return modified objects for animation optimization
        visual_objects = []
        for linkage, im in zip(self.linkages, self.linkage_im):
            visual_objects += self.draw_linkage(im, linkage.joints)
        visual_objects += self.road_im
        return visual_objects

    def visual_update(
        self, time: list[float] | float | None = None
    ) -> tuple[float, float] | None:
        """
        Update simulation and draw it.

        Parameters
        ----------
        time : list | float | None
            When a list, delta-time for physics and display (respectively)
            Using a float, only delta-time for physics, fps is set with CAMERA_SETTINGS["fps"]
            Setting to None set physics dt to pe.params["simul"]["physics_period"] and fps to CAMERA_SETTINGS["fps"]
        """
        dt: float
        fps: int
        if time is None:
            dt = pe.params["simul"]["physics_period"]
            fps = CAMERA["fps"]
        elif isinstance(time, int) or isinstance(time, float):
            dt = float(time)
            fps = CAMERA["fps"]
        else:
            dt, fps = float(time[0]), int(time[1])
        div = 1 // (dt * fps)
        update_ret: tuple[float, float] | None
        if div >= 1:
            update_list: list[float] = [0.0, 0.0]
            for _ in range(int(div)):
                result = self.update(dt)
                if result is not None:
                    for i, step_update in enumerate(result):
                        update_list[i] += step_update
            result = self.update(1 / fps - dt * div)
            if result is not None:
                for i, step_update in enumerate(result):
                    update_list[i] += step_update
            update_ret = (update_list[0], update_list[1])
        else:
            update_ret = self.update(dt)
        self.reload_visuals()
        return update_ret


def im_debug(world: VisualWorld, linkage: dynamiclinkage.DynamicLinkage) -> None:
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
    options.constraint_color = (.1, .1, .1, .0)  # type: ignore[assignment]
    world.space.debug_draw(options)


def video_debug(
    linkage: Linkage | dynamiclinkage.DynamicLinkage,
) -> None:
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


def all_linkages_video(
    linkages: list[Linkage | dynamiclinkage.DynamicLinkage],
    duration: float = 30,
    save: bool = False,
    colors: npt.NDArray[np.floating[Any]] | list[float] | list[list[float]] | None = None,
    dynamic_camera: bool = False,
) -> None:
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
    n_frames = int(CAMERA["fps"] * duration)

    dt = pe.params["simul"]["physics_period"]
    fps = CAMERA["fps"]
    if dt * fps > 1:
        print(
            f"Warning: Physics is computed every {dt}s ({1 / dt} times/s)",
            f"but display is {fps} times/s."
        )

    if colors is None:
        colors = np.logspace(0, -1, num=len(linkages))
    previous_camera = CAMERA["dynamic_camera"]
    CAMERA["dynamic_camera"] = dynamic_camera
    ani = animation.FuncAnimation(
        world.fig, world.visual_update,  # type: ignore[arg-type]
        frames=[None] * (n_frames - 1),
        init_func=partial(world.init_visuals, colors),
        interval=int(1000 / CAMERA["fps"]),
        repeat=False, blit=False
    )
    if save:
        writer = animation.FFMpegWriter(fps=CAMERA["fps"], bitrate=2500)
        ani.save(f"Dynamic {linkages[0].name}.mp4", writer=writer)
    else:
        plt.show()
        if ani:
            pass
    CAMERA["dynamic_camera"] = previous_camera


def video(
    linkage: Linkage | dynamiclinkage.DynamicLinkage,
    duration: float = 30,
    save: bool = False,
    dynamic_camera: bool = True,
) -> None:
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
    base = Static(0, 0, name="Main trick")
    crank = Crank(1, 0, name="The crank", angle=1, joint0=base)
    follower = Pivot(
        0, 2, joint0=base, joint1=crank, distance0=2, distance1=1
    )
    frame = Fixed(joint0=crank, joint1=follower, distance=1, angle=-np.pi/2)
    demo_linkage = dynamiclinkage.DynamicLinkage(
        name='Some tricky linkage',
        joints=(base, crank, follower, frame),
        space=Space()
    )
    demo_linkage.space.gravity = pe.params["physics"]["gravity"]
    video_debug(demo_linkage)
