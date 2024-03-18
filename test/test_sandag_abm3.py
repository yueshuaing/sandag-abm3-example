# ActivitySim
# See full license in LICENSE.txt.
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from activitysim.core import testing, workflow


def _example_path(dirname):
    """Paths to things in the top-level directory of this repository."""
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", dirname))


def _test_path(dirname):
    """Paths to things in the `test` directory."""
    return os.path.join(os.path.dirname(__file__), dirname)


def run_test_sandag_abm3(
    multiprocess=False, chunkless=False, sharrow=False
):

    def regress(ext):
        regress_trips_df = pd.read_csv(_test_path("regress/final_trips.csv"))
        final_trips_df = pd.read_csv(_test_path("output/final_trips.csv"))

        # column order may not match, so fix it before checking
        assert sorted(regress_trips_df.columns) == sorted(final_trips_df.columns)
        final_trips_df = final_trips_df[regress_trips_df.columns]

        # person_id,household_id,tour_id,primary_purpose,trip_num,outbound,trip_count,purpose,
        # destination,origin,destination_logsum,depart,trip_mode,mode_choice_logsum
        # compare_cols = []
        pdt.assert_frame_equal(final_trips_df, regress_trips_df)

    file_path = os.path.join(os.path.dirname(__file__), '..', "simulation.py")

    run_args = []

    if multiprocess:
        run_args.extend(
            [
                "-c",
                _test_path("configs_mp"),
                "-c",
                _example_path("configs_mp"),
            ]
        )
    elif chunkless:
        run_args.extend(
            [
                "-c",
                _test_path("configs_chunkless"),
            ]
        )
    elif sharrow:
        run_args.extend(
            [
                "-c",
                _test_path("configs_sharrow"),
            ]
        )
        if os.environ.get("GITHUB_ACTIONS") != "true":
            run_args.append("--persist-sharrow-cache")
    else:
        run_args.extend(
            [
                "-c",
                _example_path(r"configs\resident"),
            ]
        )

    out_dir = _test_path(
        f"output-{'mp' if multiprocess else 'single'}"
        f"-{'chunkless' if chunkless else 'chunked'}"
        f"-{'sharrow' if sharrow else 'no_sharrow'}"
    )

    # create output directory if it doesn't exist and add .gitignore
    Path(out_dir).mkdir(exist_ok=True)
    Path(out_dir).joinpath(".gitignore").write_text("**\n")

    run_args.extend(
        [
            "-c",
            _example_path("configs"),
            "-d",
            _example_path("data"),
            "-o",
            out_dir,
        ]
    )

    if os.environ.get("GITHUB_ACTIONS") == "true":
        subprocess.run(["coverage", "run", "-a", file_path] + run_args, check=True)
    else:
        subprocess.run([sys.executable, file_path] + run_args, check=True)

    regress()


def test_sandag_abm3():
    run_test_sandag_abm3(multiprocess=False)


def test_sandag_abm3_chunkless():
    run_test_sandag_abm3(multiprocess=False, chunkless=True)


def test_sandag_abm3_mp():
    run_test_sandag_abm3(multiprocess=True)


def test_sandag_abm3_sharrow():
    run_test_sandag_abm3(sharrow=True)


EXPECTED_MODELS = [
    "initialize_proto_population",
    "compute_disaggregate_accessibility",
    "initialize_landuse",
    "initialize_households",
    "compute_accessibility",
    "av_ownership",
    "auto_ownership_simulate",
    "work_from_home",
    "external_worker_identification",
    "external_workplace_location",
    "school_location",
    "workplace_location",
    "transit_pass_subsidy",
    "transit_pass_ownership",
    "vehicle_type_choice",
    "adjust_auto_operating_cost",
    "transponder_ownership",
    "free_parking",
    "telecommute_frequency",
    "cdap_simulate",
    "mandatory_tour_frequency",
    "mandatory_tour_scheduling",
    "school_escorting",
    "joint_tour_frequency_composition",
    "external_joint_tour_identification",
    "joint_tour_participation",
    "joint_tour_destination",
    "external_joint_tour_destination",
    "joint_tour_scheduling",
    "non_mandatory_tour_frequency",
    "external_non_mandatory_identification",
    "non_mandatory_tour_destination",
    "external_non_mandatory_destination",
    "non_mandatory_tour_scheduling",
    "vehicle_allocation",
    "tour_mode_choice_simulate",
    "atwork_subtour_frequency",
    "atwork_subtour_destination",
    "atwork_subtour_scheduling",
    "atwork_subtour_mode_choice",
    "stop_frequency",
    "trip_purpose",
    "trip_destination",
    "trip_purpose_and_destination",
    "trip_scheduling",
    "trip_mode_choice",
    "parking_location",
    "write_data_dictionary",
    "track_skim_usage",
    "write_trip_matrices",
    "write_tables",
]


@testing.run_if_exists("reference-pipeline-abm3.zip")
def test_sandag_abm3_progressive():

    import activitysim.abm  # register components # noqa: F401

    out_dir = _test_path("output-progressive")
    Path(out_dir).mkdir(exist_ok=True)
    Path(out_dir).joinpath(".gitignore").write_text("**\n")

    state = workflow.State.make_default(
        configs_dir=(
            _test_path("configs_recode"),
            _test_path("configs"),
            _example_path("resident\configs"),
        ),
        data_dir=_example_path("data"),
        # data_model_dir=_example_path("data_model"),
        output_dir=out_dir,
    )
    state.filesystem.persist_sharrow_cache()

    assert state.settings.models == EXPECTED_MODELS
    assert state.settings.chunk_size == 0
    assert not state.settings.sharrow

    for step_name in EXPECTED_MODELS:
        state.run.by_name(step_name)
        try:
            state.checkpoint.check_against(
                Path(__file__).parent.joinpath("reference-pipeline-abm3.zip"),
                checkpoint_name=step_name,
            )
        except Exception:
            print(f"> prototype_mtc_extended {step_name}: ERROR")
            raise
        else:
            print(f"> prototype_mtc_extended {step_name}: ok")


if __name__ == "__main__":
    run_test_sandag_abm3(multiprocess=False)
    run_test_sandag_abm3(multiprocess=True)
    run_test_sandag_abm3(multiprocess=False, chunkless=True)
    run_test_sandag_abm3(sharrow=True)
    test_sandag_abm3_progressive()
