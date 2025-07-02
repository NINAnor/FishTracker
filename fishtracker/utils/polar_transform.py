"""
This file is part of Fish Tracker.
Copyright 2021, VTT Technical research centre of Finland Ltd.
Developed by: Mikael Uimonen.

Fish Tracker is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Fish Tracker is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Fish Tracker.  If not, see <https://www.gnu.org/licenses/>.
"""

import math

import cv2
import numpy as np
from numba import njit

from fishtracker.utils.log_object import LogObject


@njit
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


@njit
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


@njit
def linear(x, min_x, max_x, min_y, max_y):
    return (x - min_x) / (max_x - min_x) * (max_y - min_y) + min_y


@njit
def pix2metC(y, x, cart_shape, metric_cart_shape):
    """Transforms from cartesian pixel coordinates to cartesian metric coordinates"""
    _x = linear(x, 0, cart_shape[1] - 1, 0, metric_cart_shape[1])
    _y = linear(y, 0, cart_shape[0] - 1, 0, metric_cart_shape[0])
    return (_y, _x)


@njit
def met2pixC(y, x, cart_shape, metric_cart_shape):
    """
    Transforms from cartesian metric coordinates to cartesian pixel coordinates
    """
    _x = linear(x, 0, metric_cart_shape[1], 0, cart_shape[1] - 1)
    _y = linear(y, 0, metric_cart_shape[0], 0, cart_shape[0] - 1)
    return (_y, _x)


@njit
def met2pixP(rho, phi, pol_shape, radius_limits, angle_limits):
    """Transforms from polar metric coordinates to polar pixel coordinates"""
    _rho = linear(rho, radius_limits[0], radius_limits[1], -0.5, pol_shape[0] - 0.5)
    _phi = linear(phi, *angle_limits, -0.5, pol_shape[1] - 0.5)
    return (_rho, _phi)


@njit
def pix2metP(rho, phi, pol_shape, radius_limits, angle_limits):
    """Transforms from polar pixel coordinates to polar metric coordinates"""
    _rho = linear(rho, -0.5, pol_shape[0] - 0.5, radius_limits[0], radius_limits[1])
    _phi = linear(phi, -0.5, pol_shape[1] - 0.5, *angle_limits)
    return (_rho, _phi)


@njit
def cart2polImage(
    y, x, cart_shape, metric_cart_shape, center, pol_shape, radius_limits, angle_limits
):
    """Transforms cartesian pixel coordinates to polar pixel coordinates
    by first transforming the pixel coordinates to cartesian metric coordinates,
    then to polar metric coordinates and finally to polar pixel coordinates.
    """
    _y = cart_shape[0] - (y - center[0]) - 1
    _x = center[1] - x
    y_met, x_met = pix2metC(_y, _x, cart_shape, metric_cart_shape)
    rho_met, phi_met = cart2pol(x_met, y_met)
    rho, phi = met2pixP(rho_met, phi_met, pol_shape, radius_limits, angle_limits)
    return pol_shape[0] - rho - 1, pol_shape[1] - phi - 1


@njit
def cart2polMetric(y, x, cart_shape, metric_cart_shape, center):
    """Transforms cartesian pixel coordinates to polar metric coordinates
    by first transforming the pixel coordinates to cartesian metric coordinates
    and then to polar metric coordinates.
    """
    y_met, x_met = pix2metC(y - center[0], center[1] - x, cart_shape, metric_cart_shape)
    rho_met, phi_met = cart2pol(x_met, y_met)
    return rho_met, phi_met


@njit
def createMapping(
    cart_shape, metric_cart_shape, center, pol_shape, radius_limits, angle_limits
):
    map_x = np.zeros(cart_shape, dtype=np.float32)
    map_y = np.zeros(cart_shape, dtype=np.float32)

    for j in range(cart_shape[0]):
        _j = cart_shape[0] - j - 1
        for i in range(cart_shape[1]):
            _i = cart_shape[1] - i - 1
            map_y[j, i], map_x[j, i] = cart2polImage(
                j,
                i,
                cart_shape,
                metric_cart_shape,
                center,
                pol_shape,
                radius_limits,
                angle_limits,
            )
    return map_y, map_x


class PolarTransform:
    """
    Transformes polar images to cartesian ones, based on cv2.remap mapping.
    """

    def __init__(self, pol_shape, cart_height, radius_limits, beam_angle):
        """
        Initializes the mapping function.

        Parameters:
        pol_shape -- Shape of the polar frame
        cart_height -- Height of the cartesian (output) image.
        radius_limits -- Min and max radius of the beam.
        beam_angle -- Angle covered by the beam (radians).
        """

        self.pol_shape = pol_shape
        self.cart_shape = self.getCartShape(cart_height, beam_angle)
        self.radius_limits = radius_limits
        self.angle_limits = (np.pi / 2 - beam_angle / 2, np.pi / 2 + beam_angle / 2)
        self.center = (0, (self.cart_shape[1] - 1) / 2)
        self.metric_cart_shape = (
            radius_limits[1],
            self.cart_shape[1] / self.cart_shape[0] * radius_limits[1],
        )
        self.pixels_per_meter = cart_height / radius_limits[1]

        self.map_y, self.map_x = createMapping(
            self.cart_shape,
            self.metric_cart_shape,
            self.center,
            self.pol_shape,
            self.radius_limits,
            self.angle_limits,
        )

    def getCartShape(self, height, angle):
        half_width = height * np.sin(angle / 2)
        return (height, 2 * math.ceil(half_width))

    def pix2metC(self, y, x):
        """
        Transforms from cartesian pixel coordinates to cartesian metric coordinates
        """
        return pix2metC(y, x, self.cart_shape, self.metric_cart_shape)

    def pix2metCI(self, y, x):
        """
        Transforms from cartesian pixel coordinates to cartesian metric coordinates
        (inverted y-axis)
        """
        return pix2metC(
            self.cart_shape[0] - y, x, self.cart_shape, self.metric_cart_shape
        )

    def getMetricDistance(self, y1, x1, y2, x2):
        y_met, x_met = self.pix2metC(y2 - y1, x2 - x1)
        rho_met, phi_met = cart2pol(x_met, y_met)
        return rho_met, phi_met

    def met2pixC(self, y, x):
        """
        Transforms from cartesian metric coordinates to cartesian pixel coordinates
        """
        return met2pixC(y, x, self.cart_shape, self.metric_cart_shape)

    def pix2metP(self, rho, phi):
        """Transforms from polar pixel coordinates to polar metric coordinates"""
        return pix2metP(rho, phi, self.pol_shape, self.radius_limits, self.angle_limits)

    def met2pixP(self, rho, phi):
        """Transforms from polar metric coordinates to polar pixel coordinates"""
        return met2pixP(rho, phi, self.pol_shape, self.radius_limits, self.angle_limits)

    def cart2polMetric(self, y, x, invert_y=False):
        """Transforms cartesian pixel coordinates to polar metric coordinates
        by first transforming the pixel coordinates to cartesian metric coordinates
        and then to polar metric coordinates.
        """
        if invert_y:
            return cart2polMetric(
                y - self.cart_shape[0],
                x,
                self.cart_shape,
                self.metric_cart_shape,
                self.center,
            )

        else:
            return cart2polMetric(
                y, x, self.cart_shape, self.metric_cart_shape, self.center
            )

    def cart2polImage(self, y, x):
        """Transforms cartesian pixel coordinates to polar pixel coordinates
        by first transforming the pixel coordinates to cartesian metric coordinates,
        then to polar metric coordinates and finally to polar pixel coordinates.
        """
        return cart2polImage(
            y,
            x,
            self.cart_shape,
            self.metric_cart_shape,
            self.center,
            self.pol_shape,
            self.radius_limits,
            self.angle_limits,
        )

    def pol2cartMetric(self, rho, phi, invert_y=False):
        """Transforms polar metric coordinates to cartesian pixel coordinates
        by first transforming the polar coordinates to cartesian metric coordinates,
        and then to cartesian pixel coordinates.
        """
        x_met, y_met = pol2cart(rho, phi)
        y_pix, x_pix = self.met2pixC(y_met, x_met)
        if invert_y:
            return y_pix + self.cart_shape[0] + self.center[0], self.center[1] - x_pix
        else:
            return y_pix + self.center[0], self.center[1] - x_pix

    def remap(self, image, interpolation=cv2.INTER_LINEAR):
        if not isinstance(image, np.ndarray) or image.shape != self.pol_shape:
            raise ValueError("Passed array is not of the right shape")

        return cv2.remap(image, self.map_x, self.map_y, interpolation)

    def getOuterEdge(self, distance, right=True):
        """
        Function to get the outer edge at a specific distance in cartesian pixel
        coordinates.
        Specifically built for SonarFigure to display the depth scale.
        """
        offset = np.array((0, distance if right else -distance))
        p1 = (
            np.array(
                self.pol2cartMetric(
                    -self.radius_limits[0], self.angle_limits[0 if right else 1], True
                )
            )
            + offset
        )
        p2 = (
            np.array(
                self.pol2cartMetric(
                    -self.radius_limits[1], self.angle_limits[0 if right else 1], True
                )
            )
            + offset
        )
        return np.stack((p1, p2), axis=0)


if __name__ == "__main__":

    def inverseOperations():
        pt = PolarTransform((100, 100), 100, (0, 50), np.pi / 3)
        point_c = np.array((80, 50))
        point_p = pt.cart2polMetric(*point_c)
        point_c2 = pt.pol2cartMetric(*point_p)
        LogObject().print(point_c, point_p, point_c2)

        point_p = np.array((40, 1))
        point_c = pt.pol2cartMetric(*point_p, True)
        point_p2 = pt.cart2polMetric(*point_c, True)
        LogObject().print(point_p, point_c, point_p2)

        pt.pix2metCI(200, 80)

    def mappingTest():
        polar_img_path = (
            "out/Teno1_2019-07-02_153000_polar/Teno1_2019-07-02_153000_polar_000000.png"
        )
        polar_img = cv2.imread(polar_img_path, 0)
        pt = PolarTransform((1661, 48), 1000, (2.454105, 50.837113), 0.479616)

        cv2.namedWindow("polar", 1)
        cv2.namedWindow("cartesian", 1)

        cv2.moveWindow("polar", 400, 200)
        cv2.moveWindow("cartesian", 600, 200)

        for _ in range(100):
            pt.remap(polar_img)
        cv2.imshow(
            "polar", cv2.resize(polar_img, (200, 400), interpolation=cv2.INTER_LINEAR)
        )
        cv2.imshow("cartesian", pt.remap(polar_img))

        cv2.waitKey(0)

    inverseOperations()
    mappingTest()
