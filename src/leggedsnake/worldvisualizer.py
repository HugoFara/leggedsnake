"""
This file contains class and method to see the walkers in action.

Uses Pyglet for hardware-accelerated rendering with pymunk integration.
"""
from __future__ import annotations

import warnings
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt
import pyglet
from pyglet import shapes
from pyglet.window import key
import pymunk as pm
import pymunk.pyglet_util
from pymunk import Space

# Legacy joint classes used for isinstance-based color mapping.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, message=r"pylinkage\.joints"
    )
    from pylinkage import Static, Crank, Fixed, Pivot

from pylinkage.components import Ground
from pylinkage.dyads import FixedDyad, RRRDyad
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
    "fps": 60,
}

# Type alias for bounds
Bounds = tuple[tuple[float, float], tuple[float, float]]

# Window dimensions
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Colors for different joint types (RGBA)
COLORS = {
    "static": (100, 100, 100, 255),    # Gray
    "crank": (50, 200, 50, 255),       # Green
    "fixed": (200, 50, 50, 255),       # Red
    "pivot": (50, 100, 200, 255),      # Blue
    "road": (80, 80, 80, 255),         # Dark gray
    "background": (240, 240, 245, 255), # Light gray-blue
}

# Line width for drawing
LINE_WIDTH = 3.0


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
    """Same as parent class World, but with Pyglet rendering."""

    window: pyglet.window.Window | None
    batch: pyglet.graphics.Batch
    _view_bounds: Bounds
    _scale: float
    _offset_x: float
    _offset_y: float
    _linkage_colors: list[tuple[int, int, int, int]] | None
    _running: bool
    _headless: bool

    def __init__(
        self,
        space: pm.Space | None = None,
        road_y: float = -5,
        headless: bool = False,
    ) -> None:
        """
        Instantiate the world and objects to be displayed.

        Parameters
        ----------
        space : pymunk.space.Space, optional
            Space of simulation. The default is None.
        road_y : float, optional
            The ordinate of the ground. Useful when linkages have long legs.
            The default is -5.
        headless : bool, optional
            If True, run without creating a window (for testing).
            The default is False.
        """
        super().__init__(space=space, road_y=road_y)
        self._headless = headless
        self._running = True
        self._linkage_colors = None

        # Initialize view bounds
        self._view_bounds = ((-20.0, 20.0), (-15.0, 15.0))

        # Create batch for efficient rendering
        self.batch = pyglet.graphics.Batch()

        # Set window before calling _update_scale
        if not headless:
            self.window = pyglet.window.Window(  # type: ignore[abstract]
                width=WINDOW_WIDTH,
                height=WINDOW_HEIGHT,
                caption="LeggedSnake Simulation",
                resizable=True,
            )
            self._setup_window_handlers()
        else:
            self.window = None

        self._update_scale()

    def _setup_window_handlers(self) -> None:
        """Set up Pyglet window event handlers."""
        if self.window is None:
            return

        @self.window.event
        def on_draw() -> None:
            self._on_draw()

        @self.window.event
        def on_key_press(symbol: int, modifiers: int) -> None:
            if symbol == key.ESCAPE or symbol == key.Q:
                self._running = False
                if self.window:
                    self.window.close()

        @self.window.event
        def on_resize(width: int, height: int) -> None:
            self._update_scale()

    def _update_scale(self) -> None:
        """Update the scale and offset based on current view bounds."""
        x_range = self._view_bounds[0][1] - self._view_bounds[0][0]
        y_range = self._view_bounds[1][1] - self._view_bounds[1][0]

        if self.window:
            width, height = self.window.width, self.window.height
        else:
            width, height = WINDOW_WIDTH, WINDOW_HEIGHT

        # Calculate scale to fit the view in the window
        scale_x = width / x_range if x_range > 0 else 1.0
        scale_y = height / y_range if y_range > 0 else 1.0
        self._scale = min(scale_x, scale_y) * 0.9  # 90% to leave some margin

        # Calculate offsets to center the view
        self._offset_x = width / 2 - (self._view_bounds[0][0] + x_range / 2) * self._scale
        self._offset_y = height / 2 - (self._view_bounds[1][0] + y_range / 2) * self._scale

    def _world_to_screen(self, x: float, y: float) -> tuple[float, float]:
        """Convert world coordinates to screen coordinates."""
        screen_x = x * self._scale + self._offset_x
        screen_y = y * self._scale + self._offset_y
        return screen_x, screen_y

    def _get_joint_color(self, joint: Any) -> tuple[int, int, int, int]:
        """Get the color for a joint based on its type."""
        if isinstance(joint, Crank):
            return COLORS["crank"]
        elif isinstance(joint, (Fixed, FixedDyad)):
            return COLORS["fixed"]
        elif isinstance(joint, (Pivot, RRRDyad)):
            return COLORS["pivot"]
        elif isinstance(joint, (Static, Ground)):
            return COLORS["static"]
        else:
            return COLORS["static"]

    def _on_draw(self) -> None:
        """Draw the scene."""
        if self.window is None:
            return

        self.window.clear()

        # Set background color
        pyglet.gl.glClearColor(
            COLORS["background"][0] / 255,
            COLORS["background"][1] / 255,
            COLORS["background"][2] / 255,
            1.0
        )

        # Draw road
        self._draw_road()

        # Draw linkages
        for idx, linkage in enumerate(self.linkages):
            alpha = 255
            if self._linkage_colors is not None and idx < len(self._linkage_colors):
                color = self._linkage_colors[idx]
            else:
                color = None
            self._draw_linkage(linkage, color, alpha)

        # Draw info text
        self._draw_info()

    def _draw_road(self) -> None:
        """Draw the road segments."""
        if len(self.road) < 2:
            return

        for i in range(len(self.road) - 1):
            x1, y1 = self._world_to_screen(self.road[i][0], self.road[i][1])
            x2, y2 = self._world_to_screen(self.road[i + 1][0], self.road[i + 1][1])

            line = shapes.Line(
                x1, y1, x2, y2,
                thickness=LINE_WIDTH + 1,
                color=COLORS["road"][:3],
            )
            line.draw()

    def _draw_linkage(
        self,
        linkage: dynamiclinkage.DynamicLinkage,
        override_color: tuple[int, int, int, int] | None = None,
        alpha: int = 255,
    ) -> None:
        """Draw a linkage by rendering pymunk segments and joints."""
        # Default color for bars
        bar_color = override_color[:3] if override_color else COLORS["static"][:3]

        # Draw all segments (bars) from the pymunk space that belong to this linkage
        drawn_segments: set[int] = set()
        for body in linkage.rigidbodies:
            for shape in body.shapes:
                if isinstance(shape, pm.Segment) and id(shape) not in drawn_segments:
                    drawn_segments.add(id(shape))
                    # Get world coordinates of segment endpoints
                    a = body.local_to_world(shape.a)
                    b = body.local_to_world(shape.b)
                    x1, y1 = self._world_to_screen(a.x, a.y)
                    x2, y2 = self._world_to_screen(b.x, b.y)

                    line = shapes.Line(x1, y1, x2, y2, thickness=LINE_WIDTH, color=bar_color)
                    line.opacity = alpha
                    line.draw()

        # Draw joints as circles on top
        for joint in linkage.joints:
            if override_color is not None:
                color = override_color[:3]
            else:
                color = self._get_joint_color(joint)[:3]

            sx, sy = self._world_to_screen(joint.x, joint.y)
            # Draw a filled circle for the joint
            circle = shapes.Circle(sx, sy, 5, color=color)
            circle.opacity = alpha
            circle.draw()
            # Draw a smaller inner circle for visual appeal
            inner = shapes.Circle(sx, sy, 2, color=(255, 255, 255))
            inner.opacity = alpha
            inner.draw()

    def _draw_info(self) -> None:
        """Draw information text on screen."""
        if not self.linkages or self.window is None:
            return

        center = np.mean([linkage.joints[0].coord() for linkage in self.linkages], axis=0)
        info_text = f"Position: ({int(center[0])}, {int(center[1])})  |  Press Q or ESC to quit"

        label = pyglet.text.Label(
            info_text,
            font_name='Arial',
            font_size=12,
            x=10,
            y=self.window.height - 20,
            color=(50, 50, 50, 255),
        )
        label.draw()

    def init_visuals(
        self, colors: list[float] | list[list[float]] | npt.NDArray[np.floating[Any]] | None = None
    ) -> list[Any]:
        """Initialize visual settings."""
        if colors is not None:
            processed_colors: list[tuple[int, int, int, int]] = []
            for color in colors:
                if isinstance(color, (int, float, np.floating)):
                    # Opacity value - use gray with given alpha
                    alpha = int(float(color) * 255)
                    processed_colors.append((150, 150, 150, alpha))
                else:
                    # RGB or RGBA color
                    if len(color) == 3:
                        processed_colors.append((int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), 255))
                    else:
                        processed_colors.append((int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), int(color[3] * 255)))
            self._linkage_colors = processed_colors
        return []

    def reload_visuals(self) -> list[Any]:
        """Reload the visual components only."""
        if not self.linkages:
            return []

        center = np.mean([linkage.joints[0].coord() for linkage in self.linkages], axis=0)

        prev_view: Bounds = self._view_bounds
        target: Bounds

        if CAMERA["dynamic_camera"]:
            target = (
                (float(center[0]) - 10, float(center[0]) + 10),
                (float(center[1]) - 10, float(center[1]) + 10)
            )
        else:
            target = (
                (
                    min([0.0] + [min(float(i.x) for i in linkage.joints) for linkage in self.linkages]) - 10,
                    max([0.0] + [max(float(i.x) for i in linkage.joints) for linkage in self.linkages]) + 10
                ),
                (
                    min([0.0] + [min(float(i.y) for i in linkage.joints) for linkage in self.linkages]) - 5,
                    max([0.0] + [max(float(i.y) for i in linkage.joints) for linkage in self.linkages]) + 5
                )
            )

        new_bounds = smooth_transition(target, prev_view)
        self._view_bounds = ((new_bounds[0][0], new_bounds[0][1]), (new_bounds[1][0], new_bounds[1][1]))
        self._update_scale()

        return []

    def visual_update(
        self, time: list[float] | float | None = None
    ) -> tuple[float, float] | None:
        """
        Update simulation and draw it.

        Parameters
        ----------
        time : list | float | None
            When a list, delta-time for physics and display (respectively)
            Using a float, only delta-time for physics, fps is set with CAMERA["fps"]
            Setting to None set physics dt to pe.params["simul"]["physics_period"] and fps to CAMERA["fps"]
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

    def run(self, duration: float = 30.0) -> None:
        """
        Run the simulation for the specified duration.

        Parameters
        ----------
        duration : float
            Duration in seconds to run the simulation.
        """
        if self._headless:
            # In headless mode, just run the physics
            n_frames = int(CAMERA["fps"] * duration)
            for _ in range(n_frames):
                self.visual_update()
            return

        if self.window is None:
            return

        self._running = True
        elapsed_time = 0.0
        frame_duration = 1.0 / CAMERA["fps"]

        def update(dt: float) -> None:
            nonlocal elapsed_time
            if not self._running or elapsed_time >= duration:
                pyglet.app.exit()
                return
            self.visual_update()
            elapsed_time += dt

        pyglet.clock.schedule_interval(update, frame_duration)
        pyglet.app.run()
        pyglet.clock.unschedule(update)


def im_debug(world: VisualWorld, linkage: dynamiclinkage.DynamicLinkage) -> None:
    """Use pymunk debugging for visual debugging."""
    if world.window is None:
        return

    bbox = pe.linkage_bb(linkage)
    # Update view bounds based on linkage bounding box
    world._view_bounds = (
        (float(bbox[3]) - 5, float(bbox[1]) + 5),
        (float(bbox[0]) - 5, float(bbox[2]) + 5)
    )
    world._update_scale()

    # Use pymunk's pyglet draw options
    options = pymunk.pyglet_util.DrawOptions()
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

    if world.window is None:
        return

    def update(dt: float) -> None:
        physics_dt = pe.params["simul"]["physics_period"]
        world.space.step(physics_dt)
        pe.recalc_linkage(dynamic_linkage)
        world.reload_visuals()

    pyglet.clock.schedule_interval(update, 0.2)
    pyglet.app.run()


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
        The Linkage you want to simulate. If the linkage is a Walker with
        a motor_rate attribute, that rate is used for the motor speed.
    duration : float, optional
        Duration (in seconds) of the simulation. The default is 30.
    save : bool, optional
        If you want to save it as a video file (not yet implemented for Pyglet).
    colors : list of float or list of list of float
        * If a list of float, it is the list of opacities
        * If a list of list of float, it is the list of colors
        * If None, opacities are set based on index
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

    dt = pe.params["simul"]["physics_period"]
    fps = CAMERA["fps"]
    if dt * fps > 1:
        print(
            f"Warning: Physics is computed every {dt}s ({1 / dt} times/s)",
            f"but display is {fps} times/s."
        )

    if colors is None:
        colors = list(np.logspace(0, -1, num=len(linkages)))

    previous_camera = CAMERA["dynamic_camera"]
    CAMERA["dynamic_camera"] = dynamic_camera

    world.init_visuals(colors)

    if save:
        print("Warning: Video saving is not yet implemented for Pyglet backend.")
        print("Consider using screen recording software or implementing pyglet video export.")

    world.run(duration)

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
        The Linkage you want to simulate. If the linkage is a Walker with
        a motor_rate attribute, that rate is used for the motor speed.
    duration : float, optional
        Duration (in seconds) of the simulation. The default is 30.
    save : bool, optional
        If you want to save it as a video file (not yet implemented for Pyglet).
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
