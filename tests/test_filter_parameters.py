import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath("."))

from fishtracker.core.processing.batch_track import BatchTrack
from fishtracker.core.tracking.tracker import AllTrackerParameters, TrackerParameters
from fishtracker.parameters.detector_parameters import DetectorParameters
from fishtracker.parameters.filter_parameters import FilterParameters


def test_filter_parameters():
    """Test that filter parameters are properly applied in batch processing."""

    with tempfile.TemporaryDirectory() as temp_dir:
        primary_params = TrackerParameters(
            max_age=10, min_hits=5, search_radius=10, trim_tails=True
        )
        filter_params = FilterParameters(min_duration=5, mad_limit=10)
        secondary_params = TrackerParameters(
            max_age=10, min_hits=5, search_radius=10, trim_tails=True
        )

        tracker_params = AllTrackerParameters(
            primary_params, filter_params, secondary_params
        )
        detector_params = DetectorParameters()

        # test 1: export_all_tracks=True (should ignore filters)
        batch_track_all = BatchTrack(
            display=False,
            files=[],
            save_directory=temp_dir,
            parallel=1,
            params_detector=detector_params,
            params_tracker=tracker_params,
            export_all_tracks=True,
        )

        # test 2: export_all_tracks=False (should apply filters)
        batch_track_filtered = BatchTrack(
            display=False,
            files=[],
            save_directory=temp_dir,
            parallel=1,
            params_detector=detector_params,
            params_tracker=tracker_params,
            export_all_tracks=False,
        )

        assert batch_track_all.export_all_tracks is True
        assert batch_track_filtered.export_all_tracks is False

        assert (
            batch_track_all.tracker_params.filter.getParameter(
                FilterParameters.ParametersEnum.min_duration
            )
            == 5
        )
        assert (
            batch_track_all.tracker_params.filter.getParameter(
                FilterParameters.ParametersEnum.mad_limit
            )
            == 10
        )
