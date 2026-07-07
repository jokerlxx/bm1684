import time

from app.pipeline.job_scheduler import InferenceJob, JobTable, group_single_model_batches, pick_jobs_for_tick
from app.pipeline.model_scheduler import DETECTOR_INPUTS


def test_job_table_rebuild_creates_jobs_per_stream():
    table = JobTable()
    table.rebuild(
        enabled_detectors={"fight", "fall"},
        detector_streams={"fight": {0, 2}, "fall": {1}},
        detector_inputs=DETECTOR_INPUTS,
        interval_for_detector=lambda name: 0.1 if name == "fight" else 0.2,
        now=0.0,
    )
    jobs = table.active_jobs()
    assert len(jobs) == 3
    assert table.job_stream_ids() == {0, 1, 2}


def test_job_table_disable_removes_jobs():
    table = JobTable()
    table.rebuild(
        enabled_detectors={"fight"},
        detector_streams={"fight": {0}},
        detector_inputs=DETECTOR_INPUTS,
        interval_for_detector=lambda _: 0.1,
        now=0.0,
    )
    table.rebuild(
        enabled_detectors=set(),
        detector_streams={},
        detector_inputs=DETECTOR_INPUTS,
        interval_for_detector=lambda _: 0.1,
        now=0.0,
    )
    assert table.active_jobs() == []


def test_due_jobs_and_mark_served():
    table = JobTable()
    table.rebuild(
        enabled_detectors={"fight"},
        detector_streams={"fight": {0}},
        detector_inputs=DETECTOR_INPUTS,
        interval_for_detector=lambda _: 0.5,
        now=0.0,
    )
    job = table.active_jobs()[0]
    assert table.due_jobs(0.0)
    table.mark_served(job, 0.0)
    assert not table.due_jobs(0.1)
    assert table.due_jobs(0.5)


def test_group_single_model_batches_merges_same_model():
    fight_a = InferenceJob(0, "fight", ["fight_detection"], 0.1, 10)
    fight_b = InferenceJob(2, "fight", ["fight_detection"], 0.1, 10)
    fall = InferenceJob(1, "fall", ["fall_detection"], 0.1, 10)
    ready = [(fight_a, object()), (fall, object()), (fight_b, object())]
    batches = group_single_model_batches(ready, lambda key: 9 if key == "fight_detection" else 1)
    assert len(batches) == 2
    fight_batch = next(items for key, items in batches if key == "fight_detection")
    assert len(fight_batch) == 2
    assert {job.stream_id for job, _ in fight_batch} == {0, 2}


def test_pick_jobs_prefers_higher_priority():
    fight = InferenceJob(0, "fight", ["fight_detection"], 0.1, 10)
    window = InferenceJob(1, "window_door_inside", ["window_door_inside"], 1.0, 3)
    picked = pick_jobs_for_tick([window, fight], max_jobs=1)
    assert picked[0].detector_name == "fight"
