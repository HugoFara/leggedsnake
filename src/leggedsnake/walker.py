#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The walker module gives a simple interface between a generic linkage and a Walker.

A generic linkage should be understood as a pylinkage.linkage.Linkage. Thus a
Walker is a kinematic mechanism (not dynamic).

Created on Thu Jun 10 2021 21:13:12.

@author: HugoFara
"""
from __future__ import annotations

from math import tau

import pylinkage.linkage as lk
from pylinkage import Static, Crank, Fixed, Pivot, Revolute
from pylinkage.joints.joint import Joint


class Walker(lk.Linkage):  # type: ignore[misc]
    """A Walker, or a leg mechanism is a Linkage with some useful methods."""

    def add_legs(self, number: int = 2) -> None:
        """
        Add legs to a linkage, mostly for a dynamic simulation.

        The leg is a subset a joints whose position inherits from a crank.

        Parameters
        ----------
        number : int, optional
            Number of legs to add. The default is 2.

        Returns
        -------
        None.

        """
        new_joints: list[Joint] = []
        iterations_factor = int(12 / (number + 1)) + 1
        # We use at least 12 steps to avoid bad initial positions
        new_positions = tuple(
            self.step(
                (number + 1) * iterations_factor,
                self.get_rotation_period() / (number + 1) / iterations_factor
            )
        )[iterations_factor - 1:-1:iterations_factor]
        # Because we have per-leg iterations,
        # we have to save crank information
        crank_memory: dict[Joint, Joint] = dict(zip(self._cranks, self._cranks))
        # For each leg
        for i, positions in enumerate(new_positions):
            equiv: dict[Joint | None, Joint | None] = {None: None}
            # For each new joint
            new_j: Joint
            for pos, j in zip(positions, self._solve_order):
                if isinstance(j.joint0, Static) and j.joint0 not in equiv:
                    equiv[j.joint0] = j.joint0
                common: dict[str, float | Joint | None | str] = {
                    'x': pos[0], 'y': pos[1],
                    'joint0': equiv[j.joint0],
                    'name': j.name + ' ({})'.format(i)
                }
                if isinstance(j, Static):
                    new_j = j
                elif isinstance(j, Crank):
                    common['joint1'] = crank_memory[j]
                    new_j = Fixed(
                        **common,
                        distance=j.r,
                        angle=tau / (number + 1)
                    )
                    crank_memory[j] = new_j
                    new_joints.append(new_j)
                else:
                    # Static joints not always included in joints
                    if isinstance(j.joint1, Static) and j.joint1 not in equiv:
                        equiv[j.joint1] = j.joint1
                    common['joint1'] = equiv[j.joint1]

                    if isinstance(j, Fixed):
                        new_j = Fixed(
                            **common,
                            distance=j.r, angle=j.angle
                        )
                    elif isinstance(j, Revolute):
                        new_j = Revolute(
                            **common,
                            distance0=j.r0, distance1=j.r1
                        )
                    new_joints.append(new_j)
                equiv[j] = new_j
        self.joints += tuple(new_joints)
        self._solve_order += tuple(new_joints)

    def get_foots(self) -> list[Joint]:
        """
        Return the list of foot joints, joints that should be used as foots.

        Formally, they are joints without children; their positions do not
        influence other joints.

        Returns
        -------
        candidates : list
            List of terminal joints, feet.

        """
        candidates = list(self.joints)
        for j in self.joints:
            if j.joint0 in candidates:
                candidates.remove(j.joint0)
            if hasattr(j, 'joint1') and j.joint1 in candidates:
                candidates.remove(j.joint1)
        return candidates

    def mirror_leg(self, axis_x: float = 0.0) -> None:
        """
        Create a mirrored copy of the leg across a vertical axis.

        This is useful for creating symmetric walkers with left and right legs.
        The mirrored leg has its X-coordinates reflected and angles negated
        to produce a geometrically mirrored version.

        Unlike add_legs() which creates phase-offset copies along the same axis,
        mirror_leg() creates a symmetric copy on the opposite side of the body.

        Parameters
        ----------
        axis_x : float, optional
            X-coordinate of the vertical mirror axis. The default is 0.0,
            which mirrors across the Y-axis.

        Returns
        -------
        None.

        Examples
        --------
        >>> walker = create_single_leg_linkage()
        >>> walker.mirror_leg()  # Creates symmetric left/right pair
        >>> walker.add_legs(1)   # Add phase-offset copies of both legs
        """
        new_joints: list[Joint] = []
        new_cranks: list[Crank] = []
        # Map original joints to their mirrored counterparts
        equiv: dict[Joint | None, Joint | None] = {None: None}

        for j in self._solve_order:
            # Mirror X coordinate across the axis
            mirrored_x = 2 * axis_x - j.x

            common: dict[str, float | Joint | None | str] = {
                'x': mirrored_x,
                'y': j.y,
                'joint0': equiv.get(j.joint0, j.joint0),
                'name': j.name + ' (mirrored)'
            }

            if isinstance(j, Static):
                # Static joints are shared frame points; create mirrored copy
                new_j = Static(
                    x=mirrored_x, y=j.y,
                    name=j.name + ' (mirrored)'
                )
                # Preserve drawing linkage
                if hasattr(j, 'joint0') and j.joint0 is not None:
                    new_j.joint0 = equiv.get(j.joint0, j.joint0)
                new_joints.append(new_j)
            elif isinstance(j, Crank):
                # Crank rotation stays the same; initial position is mirrored
                new_j = Crank(
                    **common,
                    distance=j.r,
                    angle=j.angle,  # Keep same rotation direction for walking
                )
                new_joints.append(new_j)
                new_cranks.append(new_j)
            elif isinstance(j, Fixed):
                # Mirror angle by negating it
                common['joint1'] = equiv.get(j.joint1, j.joint1)
                new_j = Fixed(
                    **common,
                    distance=j.r,
                    angle=-j.angle  # Negate angle for mirror effect
                )
                new_joints.append(new_j)
            elif isinstance(j, Revolute):
                common['joint1'] = equiv.get(j.joint1, j.joint1)
                new_j = Revolute(
                    **common,
                    distance0=j.r0,
                    distance1=j.r1
                )
                new_joints.append(new_j)
            else:
                # Unknown joint type, try generic copy
                continue

            equiv[j] = new_j

        self.joints += tuple(new_joints)
        self._solve_order += tuple(new_joints)
        # Also update cranks list for compatibility with add_legs
        self._cranks = tuple(list(self._cranks) + new_cranks)
