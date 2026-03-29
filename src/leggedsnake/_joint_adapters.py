"""
Adapter classes bridging pylinkage's modern API to the legacy attribute names.

DynamicJoint subclasses (Nail, PinUp, DynamicPivot, Motor) need attributes
like ``joint0``, ``joint1``, ``r``, ``set_anchor0()`` that the legacy
``pylinkage.joints`` classes provided.  These adapters inherit from the
**new** API (Ground, FixedDyad, RRRDyad, actuators.Crank) and add
properties/methods that map to the old names.

When pylinkage 1.0 removes the legacy module, only this file needs updating.
"""
from __future__ import annotations

from typing import Any

from pylinkage.components import Ground, ConnectedComponent, _AnchorProxy
from pylinkage.dyads import FixedDyad, RRRDyad
from pylinkage.actuators import Crank as ActuatorCrank


class GroundAdapter(Ground):
    """Ground with legacy Static-compatible interface."""

    __slots__ = ("_compat_joint0", "_compat_joint1")

    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        name: str | None = None,
    ) -> None:
        super().__init__(x, y, name)
        self._compat_joint0: Any = None
        self._compat_joint1: Any = None

    @property
    def joint0(self) -> Any:
        return self._compat_joint0

    @joint0.setter
    def joint0(self, value: Any) -> None:
        self._compat_joint0 = value

    @property
    def joint1(self) -> Any:
        return self._compat_joint1

    @joint1.setter
    def joint1(self, value: Any) -> None:
        self._compat_joint1 = value


class FixedDyadAdapter(FixedDyad):
    """FixedDyad with legacy Fixed-compatible interface.

    Provides ``joint0``/``joint1``/``r`` property aliases and
    ``set_anchor0``/``set_anchor1`` methods.
    """

    __slots__ = ()

    def __init__(
        self,
        x: float | None = None,
        y: float | None = None,
        joint0: Any = None,
        joint1: Any = None,
        distance: float | None = None,
        angle: float | None = None,
        name: str | None = None,
        **_kwargs: Any,
    ) -> None:
        dist = distance if distance is not None else 0.0
        ang = angle if angle is not None else 0.0
        # Bypass FixedDyad.__init__ which calls _initialize_position
        # (parents may not be positioned yet during DynamicJoint construction)
        ConnectedComponent.__init__(self, x, y, name)
        self.anchor1 = joint0
        self.anchor2 = joint1
        self.distance = dist
        self.angle = ang

    # --- Legacy property aliases ---

    @property
    def joint0(self) -> Any:
        return self.anchor1

    @joint0.setter
    def joint0(self, value: Any) -> None:
        self.anchor1 = value

    @property
    def joint1(self) -> Any:
        return self.anchor2

    @joint1.setter
    def joint1(self, value: Any) -> None:
        self.anchor2 = value

    @property
    def r(self) -> float:
        return self.distance

    @r.setter
    def r(self, value: float) -> None:
        self.distance = value

    # --- Legacy methods ---

    def set_anchor0(
        self,
        joint: Any,
        distance: float | None = None,
        angle: float | None = None,
    ) -> None:
        """Legacy Fixed.set_anchor0 equivalent."""
        self.anchor1 = joint
        if distance is not None:
            self.distance = distance
        if angle is not None:
            self.angle = angle

    def set_anchor1(self, joint: Any) -> None:
        """Legacy Fixed.set_anchor1 equivalent."""
        self.anchor2 = joint


class RRRDyadAdapter(RRRDyad):
    """RRRDyad with legacy Pivot/Revolute-compatible interface.

    Provides ``joint0``/``joint1``/``r0``/``r1`` property aliases and
    ``set_anchor0``/``set_anchor1`` methods.
    """

    __slots__ = ()

    def __init__(
        self,
        x: float | None = None,
        y: float | None = None,
        joint0: Any = None,
        joint1: Any = None,
        distance0: float | None = None,
        distance1: float | None = None,
        name: str | None = None,
        **_kwargs: Any,
    ) -> None:
        d1 = distance0 if distance0 is not None else 0.0
        d2 = distance1 if distance1 is not None else 0.0
        # Bypass RRRDyad.__init__ which may call _initialize_position
        ConnectedComponent.__init__(self, x, y, name)
        self.anchor1 = joint0
        self.anchor2 = joint1
        self.distance1 = d1
        self.distance2 = d2

    # --- Legacy property aliases ---

    @property
    def joint0(self) -> Any:
        return self.anchor1

    @joint0.setter
    def joint0(self, value: Any) -> None:
        self.anchor1 = value

    @property
    def joint1(self) -> Any:
        return self.anchor2

    @joint1.setter
    def joint1(self, value: Any) -> None:
        self.anchor2 = value

    @property
    def r0(self) -> float:
        return self.distance1

    @r0.setter
    def r0(self, value: float) -> None:
        self.distance1 = value

    @property
    def r1(self) -> float:
        return self.distance2

    @r1.setter
    def r1(self, value: float) -> None:
        self.distance2 = value

    # --- Legacy methods ---

    def set_anchor0(self, joint: Any, distance: float | None = None) -> None:
        """Legacy Pivot.set_anchor0 equivalent."""
        self.anchor1 = joint
        if distance is not None:
            self.distance1 = distance

    def set_anchor1(self, joint: Any, distance: float | None = None) -> None:
        """Legacy Pivot.set_anchor1 equivalent."""
        self.anchor2 = joint
        if distance is not None:
            self.distance2 = distance


class CrankAdapter(ActuatorCrank):
    """actuators.Crank with legacy Crank-compatible interface.

    Provides ``joint0``/``r``/``angle`` property aliases and
    ``set_anchor0`` method. Accepts legacy-style constructor arguments.
    """

    __slots__ = ()

    def __init__(
        self,
        x: float | None = None,
        y: float | None = None,
        joint0: Any = None,
        distance: float | None = None,
        angle: float | None = None,
        name: str | None = None,
        **_kwargs: Any,
    ) -> None:
        # Bypass ActuatorCrank.__init__ which requires a positioned Ground
        # and computes output position internally. During Motor construction,
        # joint0 is a DynamicJoint (Nail), not a Ground.
        ConnectedComponent.__init__(self, x, y, name)
        self.anchor = joint0  # type: ignore[assignment]
        self.radius = distance if distance is not None else 0.0
        self.angular_velocity = angle if angle is not None else 0.0
        self.initial_angle = 0.0
        self._output = _AnchorProxy(self)
        self._omega = None
        self._alpha = None

    # --- Legacy property aliases ---

    @property
    def joint0(self) -> Any:
        return self.anchor

    @joint0.setter
    def joint0(self, value: Any) -> None:
        self.anchor = value

    @property
    def r(self) -> float:
        return self.radius

    @r.setter
    def r(self, value: float) -> None:
        self.radius = value

    @property
    def angle(self) -> float:  # type: ignore[override]
        return self.angular_velocity

    @angle.setter
    def angle(self, value: float) -> None:
        self.angular_velocity = value

    # --- Legacy methods ---

    def set_anchor0(self, joint: Any, distance: float | None = None) -> None:
        """Legacy Crank.set_anchor0 equivalent."""
        self.anchor = joint
        if distance is not None:
            self.radius = distance

    def reload(self, dt: float = 1) -> None:
        """Advance the crank, tolerating non-Ground anchors."""
        from pylinkage.solver.joints import solve_crank

        anchor = self.anchor
        ax = getattr(anchor, 'x', None)
        ay = getattr(anchor, 'y', None)
        if ax is None or ay is None or self.x is None or self.y is None:
            return

        self.x, self.y = solve_crank(
            self.x, self.y,
            ax, ay,
            self.radius,
            self.angular_velocity,
            dt,
        )
