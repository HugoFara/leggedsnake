#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The walker module give a simple interface between a generic linkage and a Walker.

A generic linkage should be understood as a pylinkage.linkage.Linkage. Thus a
Walker is a kinematic mechanism (not dynamic).

Created on Thu Jun 10 21:13:12 2021.

@author: HugoFara
"""
from math import tau
import pylinkage.linkage as lk


class Walker(lk.Linkage):
    """A Walker, or a leg mechanism, is a Linkage with some useful methods."""

    def add_legs(self, number=2):
        """
        Add legs to a linkage, mostly for a dynamic simulation.

        The leg is a subset a joints whose position inheritates from a crank.

        Parameters
        ----------
        number : int, optional
            Number of legs to add. The default is 2.

        Returns
        -------
        None.

        """
        new_joints = []
        iterations_factor = int(12 / (number + 1)) + 1
        # We use at least 12 steps to avoid bad initial positions
        new_positions = tuple(
            self.step(
                (number + 1) * iterations_factor,
                self.get_rotation_period() / (number + 1) / iterations_factor
            )
        )[iterations_factor - 1:-1:iterations_factor]
        # Because we have per-leg iterations
        # we have to save crank informations
        crank_memory = dict(zip(self._cranks, self._cranks))
        # For each leg
        for i, positions in enumerate(new_positions):
            equiv = {None: None}
            # For each new joint
            for pos, j in zip(positions, self._solve_order):
                if isinstance(j.joint0, lk.Static) and j.joint0 not in equiv:
                    equiv[j.joint0] = j.joint0
                common = {
                    'x': pos[0], 'y': pos[1],
                    'joint0': equiv[j.joint0],
                    'name': j.name + ' ({})'.format(i)
                }
                if isinstance(j, lk.Static):
                    new_j = j
                elif isinstance(j, lk.Crank):
                    common['joint1'] = crank_memory[j]
                    new_j = lk.Fixed(
                        **common, distance=j.r,
                        angle=tau / (number + 1)
                    )
                    crank_memory[j] = new_j
                    new_joints.append(new_j)
                else:
                    # Static joints not always included in joints
                    if isinstance(j.joint1, lk.Static) and j.joint1 not in equiv:
                        equiv[j.joint1] = j.joint1
                    common['joint1'] = equiv[j.joint1]

                    if isinstance(j, lk.Fixed):
                        new_j = lk.Fixed(
                            **common, distance=j.r, angle=j.angle
                        )
                    elif isinstance(j, lk.Pivot):
                        new_j = lk.Pivot(
                            **common, distance0=j.r0, distance1=j.r1
                        )
                    new_joints.append(new_j)
                equiv[j] = new_j
        self.joints += tuple(new_joints)
        self._solve_order += tuple(new_joints)

    def get_foots(self):
        """
        Return the list of foot joints, joints that should be used as foots.

        Formally, they are joints without children, there position does not
        influence other joints.

        Returns
        -------
        candidates : list
            List of terminal joints, foots.

        """
        candidates = list(self.joints)
        for j in self.joints:
            if j.joint0 in candidates:
                candidates.remove(j.joint0)
            if hasattr(j, 'joint1') and j.joint1 in candidates:
                candidates.remove(j.joint1)
        return candidates
