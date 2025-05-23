"""
SORT: A Simple, Online and Realtime Tracker
Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

---------------------------------------------------------------------------

This file is a modified version of the original and used as part of Fish Tracker.
Copyright 2021, VTT Technical research centre of Finland Ltd.
Developed by: Otto Korkalo and Mikael Uimonen

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

import matplotlib
import numpy as np

matplotlib.use("Agg")

from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y, strict=False)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IUO between two bboxes in the form [l,t,w,h]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )
    return o


def eucl_batch(bb_test, bb_gt, radius=10):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    x1 = bb_test[..., 0]
    y1 = bb_test[..., 1]
    x2 = bb_gt[..., 0]
    y2 = bb_gt[..., 1]

    o = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)

    o[o > radius**2] = 10e100  # TODO

    return o


class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as
    bbox.
    """

    count = 0

    def __init__(self, z):
        KalmanBoxTracker.count += 1

        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.id = KalmanBoxTracker.count

        dt = 0.1
        self.kf.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

        self.kf.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

        self.kf.R *= 1
        self.kf.P *= 1000.0
        self.kf.Q *= 0.1

        self.kf.x[0] = z[0]
        self.kf.x[2] = z[1]

        self.time_since_update = 0
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.last_det_frame = 0
        self.last_det_ind = -1
        self.status = 0  # 0 = candidate, 1 = active, 2 = lost, 3 = removed
        self.search_radius_coeff = 1

    def update(self, z):
        self.kf.update(z)

    def predict(self):
        self.kf.predict()
        x0 = float(self.kf.x[0])
        x2 = float(self.kf.x[2])
        self.history.append([x0, x2])
        return self.history[-1]

    def get_state(self):
        return np.array([self.kf.x[0], self.kf.x[2]])

    def get_status(self):
        return self.status

    def set_status(self, status):
        self.status = status

    def get_hit_streak(self):
        return self.hit_streak


def associate_detections_to_trackers(detections, trackers, search_radius=10):
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 2), dtype=int),
        )

    cost_matrix = eucl_batch(detections, trackers, search_radius)

    if min(cost_matrix.shape) > 0:
        matched_indices = linear_assignment(cost_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, _ in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, _ in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out matched with too high cost
    matches = []
    for m in matched_indices:
        if cost_matrix[m[0], m[1]] > search_radius**2 * 100000000:  # TODO
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort:
    def __init__(self, max_age=1, min_hits=3, search_radius=10):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.search_radius = search_radius

        self.trackers = []
        self.frame_count = 0

    def update(self, detz=np.empty((0, 4))):
        dets = np.empty((0, 2))
        if len(detz) > 0:
            dets = np.empty((len(detz), 2))
            dets[:, 0] = detz[:, 0] + (detz[:, 2] - detz[:, 0]) / 2.0
            dets[:, 1] = detz[:, 1] + (detz[:, 3] - detz[:, 1]) / 2.0

        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 2))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            self.trackers[t].time_since_update += 1

            # Existing track set to lost
            if self.trackers[t].get_status() == 1:
                self.trackers[t].set_status(2)

            # Do not initiate tracks withtout consecutive associations
            if (
                self.trackers[t].get_status() == 0
                and self.trackers[t].time_since_update > 0
            ):
                self.trackers[t].time_since_update = self.max_age + 1

            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1]]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # TODO alla oleva turha?
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.search_radius
        )

        for m in matched:
            tr = self.trackers[m[1]]
            tr.update(dets[m[0], :])

            tr.time_since_update = 0
            tr.hit_streak += 1

            tr.last_det_ind = m[0]
            tr.last_det_frame = self.frame_count

            if tr.get_status() == 0 and tr.get_hit_streak() > self.min_hits:
                tr.set_status(1)

            if tr.get_status() == 2:
                tr.set_status(1)

        for m in unmatched_trks:
            tr = self.trackers[m]
            if tr.get_status() == 2:
                tr.hit_streak = 0

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            # do not initialize new track to existing bounding box area
            new_allowed = True
            for trk in self.trackers:
                bb = np.array([trk.get_state()[0], trk.get_state()[1]]).reshape(1, -1)
                cost = eucl_batch(bb, [dets[i, :]], self.search_radius)
                if cost[0, 0] < self.search_radius**2:
                    new_allowed = False

            if new_allowed:
                trk = KalmanBoxTracker(dets[i, :])
                trk.last_det_ind = i
                trk.last_det_frame = self.frame_count
                self.trackers.append(trk)

        i = len(self.trackers)

        for trk in reversed(self.trackers):
            if True:
                d = np.array([trk.get_state()[0], trk.get_state()[1]])
                d_ind = (
                    trk.last_det_ind if trk.last_det_frame == self.frame_count else -1
                )
                status = trk.get_status()
                # HACK: "lost" are treated as "tracked",
                # this is for main application use
                if status == 2:
                    status = 1

                ret.append(
                    np.concatenate(
                        (
                            d[0] - 10,
                            d[1] - 10,
                            d[0] + 10,
                            d[1] + 10,
                            [trk.id],
                            [status],
                            [trk.get_hit_streak()],
                            [d_ind],
                            [self.search_radius],
                        )
                    ).reshape(1, -1)
                )
            i -= 1

            # Delete tracks
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                # trk.status = 3

        if len(ret) > 0:
            return np.concatenate(ret)

        return np.empty((0, 8))
