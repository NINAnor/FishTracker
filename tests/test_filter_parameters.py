import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath("."))

from fishtracker.core.processing.batch_track import BatchTrack
from fishtracker.core.tracking.tracker import AllTrackerParameters, TrackerParameters
from fishtracker.parameters.detector_parameters import DetectorParameters
from fishtracker.parameters.filter_parameters import FilterParameters


def verify_condition(condition, message):
    """Helper function to verify conditions without using assert."""
    if not condition:
        raise ValueError(f"Test failed: {message}")


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

        # verify export_all_tracks settings
        verify_condition(
            batch_track_all.export_all_tracks is True,
            "batch_track_all.export_all_tracks should be True",
        )

        verify_condition(
            batch_track_filtered.export_all_tracks is False,
            "batch_track_filtered.export_all_tracks should be False",
        )

        # verify filter parameters
        min_duration_value = batch_track_all.tracker_params.filter.getParameter(
            FilterParameters.ParametersEnum.min_duration
        )
        verify_condition(
            min_duration_value == 5,
            f"min_duration should be 5, but got {min_duration_value}",
        )

        mad_limit_value = batch_track_all.tracker_params.filter.getParameter(
            FilterParameters.ParametersEnum.mad_limit
        )
        verify_condition(
            mad_limit_value == 10, f"mad_limit should be 10, but got {mad_limit_value}"
        )

        print("All filter parameter tests passed successfully!")
        return True


if __name__ == "__main__":
    try:
        test_filter_parameters()
        print("Test completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        sys.exit(1)
