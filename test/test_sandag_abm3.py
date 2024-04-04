# ActivitySim
# See full license in LICENSE.txt.
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from activitysim.core import workflow


def _example_path(dirname):
    """Paths to things in the top-level directory of this repository."""
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", dirname))


def _test_path(dirname) -> Path:
    """Paths to things in the `test` directory."""
    return Path(__file__).parent.joinpath(dirname)


def regress(out_dir: Path, regress_dir: Path = None):
    if regress_dir is None:
        regress_dir = _test_path("regress")
    regress_trips_df = pd.read_csv(regress_dir.joinpath("final_trips.csv"))
    final_trips_df = pd.read_csv(out_dir / "final_trips.csv")

    # columns that are in the regression target must be in the output
    missing_columns = set(regress_trips_df.columns) - set(final_trips_df.columns)
    assert missing_columns == set()

    # column order may not match, so fix it before checking
    final_trips_df = final_trips_df[regress_trips_df.columns]

    pdt.assert_frame_equal(final_trips_df, regress_trips_df)


def run_test_sandag_abm3(multiprocess=False, chunkless=False, sharrow=False):
    file_path = os.path.join(os.path.dirname(__file__), "..", "simulation.py")

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
                _example_path(r"configs/resident"),
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

    regress(out_dir, _test_path("regress/hh500"))


def test_sandag_abm3_cli():
    run_test_sandag_abm3(multiprocess=False)


def test_sandag_abm3():
    from activitysim import abm  # noqa: F401

    # make a temp directory for output that will persist for review
    tmp_path = Path(__file__).parent.joinpath("output-test_sandag_abm3")
    tmp_path.mkdir(exist_ok=True)

    state = workflow.State.make_default(
        configs_dir=(
            _example_path(r"configs/common"),
            _example_path(r"configs/resident"),
        ),
        data_dir=_example_path("data"),
        output_dir=tmp_path,
        settings=dict(
            cleanup_pipeline_after_run=False,
            # recode_pipeline_columns=True,
            treat_warnings_as_errors=True,
            households_sample_size=100,
            # households_sample_size=500, == 12min 43 sec
            chunk_size=0,
            use_shadow_pricing=True,
        ),
    )
    state.import_extensions("../extensions")
    state.filesystem.persist_sharrow_cache()
    state.run.all(resume_after=None)
    regress(tmp_path, _test_path("regress/hh500"))


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


@pytest.mark.parametrize("use_sharrow", [False, True])
def test_sandag_abm3_progressive(use_sharrow):
    import activitysim.abm  # register components # noqa: F401

    out_dir = _test_path("output-progressive")
    out_dir.mkdir(exist_ok=True)
    out_dir.joinpath(".gitignore").write_text("**\n")

    settings = dict(
        cleanup_pipeline_after_run=False,
        treat_warnings_as_errors=True,
        households_sample_size=500,
        chunk_size=0,
        use_shadow_pricing=True,
    )
    tags = ["-hh500"]

    if use_sharrow:
        settings["sharrow"] = "test"
        settings["recode_pipeline_columns"] = True
        tags.append("-recode")

    state = workflow.State.make_default(
        configs_dir=(
            _example_path(r"configs/common"),
            _example_path(r"configs/resident"),
        ),
        data_dir=_example_path("data"),
        output_dir=out_dir,
        settings=settings,
    )
    state.import_extensions("../extensions")
    state.filesystem.persist_sharrow_cache()
    state.logging.config_logger()

    assert state.settings.models == EXPECTED_MODELS
    assert state.settings.chunk_size == 0
    if not use_sharrow:
        assert not state.settings.sharrow

    tags = "".join(tags)
    ref_pipeline = Path(__file__).parent.joinpath(
        f"regress/reference-pipeline{tags}.zip"
    )

    for step_name in EXPECTED_MODELS:
        state.run.by_name(step_name)
        try:
            if ref_pipeline.exists():
                state.checkpoint.check_against(ref_pipeline, checkpoint_name=step_name)
                pass
        except Exception:
            print(f"> sandag-abm3 {step_name}: ERROR")
            raise
        else:
            print(f"> sandag-abm3 {step_name}: ok")


if __name__ == "__main__":
    # run_test_sandag_abm3(multiprocess=True)
    test_sandag_abm3_progressive(use_sharrow=False)
    test_sandag_abm3_progressive(use_sharrow=True)
