import pytest

from shoprl.platform import (TERMINAL, InvalidTransition, Job, JobState,
                             JobStore, can_transition)
from shoprl.platform.store import ConcurrentModification, JobNotFound

S = JobState


def test_create_is_pending_and_persisted(tmp_path):
    s = JobStore(tmp_path / "j.db")
    job = s.create("rollout", {"prompt_id": "P-1"})
    assert job.state is S.PENDING and job.attempts == 0
    assert s.get(job.id).payload == {"prompt_id": "P-1"}


def test_valid_lifecycle(tmp_path):
    s = JobStore(tmp_path / "j.db")
    j = s.create("optimize")
    assert s.transition(j.id, S.RUNNING).state is S.RUNNING
    assert s.transition(j.id, S.SUCCEEDED).state is S.SUCCEEDED


def test_invalid_transitions_rejected(tmp_path):
    s = JobStore(tmp_path / "j.db")
    j = s.create("reward")
    with pytest.raises(InvalidTransition):          # can't skip RUNNING
        s.transition(j.id, S.SUCCEEDED)
    s.transition(j.id, S.RUNNING)
    s.transition(j.id, S.SUCCEEDED)
    with pytest.raises(InvalidTransition):          # can't resurrect a terminal job
        s.transition(j.id, S.RUNNING)


def test_retry_path_and_dead_letter(tmp_path):
    s = JobStore(tmp_path / "j.db")
    j = s.create("rollout", max_attempts=2)
    s.transition(j.id, S.RUNNING)
    s.transition(j.id, S.FAILED, error="boom", bump_attempt=True)
    j2 = s.transition(j.id, S.RETRYING)
    assert j2.attempts == 1
    s.transition(j.id, S.PENDING)                   # requeued
    s.transition(j.id, S.RUNNING)
    s.transition(j.id, S.FAILED, bump_attempt=True) # attempts now 2 == max
    dead = s.transition(j.id, S.DEAD_LETTER)
    assert dead.state is S.DEAD_LETTER and dead.attempts == 2


def test_terminal_states_have_no_exits():
    for t in TERMINAL:
        assert all(not can_transition(t, other) for other in JobState)


def test_survives_restart(tmp_path):
    path = tmp_path / "jobs.db"
    s1 = JobStore(path)
    a = s1.create("rollout"); b = s1.create("optimize")
    s1.transition(a.id, S.RUNNING); s1.transition(a.id, S.SUCCEEDED)
    s1.close()                                      # process "crash"/restart
    s2 = JobStore(path)                             # reopen the same DB file
    assert s2.get(a.id).state is S.SUCCEEDED        # in-flight state recovered
    assert s2.get(b.id).state is S.PENDING
    assert s2.counts() == {"succeeded": 1, "pending": 1}


def test_atomic_guard_detects_lost_race(tmp_path):
    # Two workers race to claim the same PENDING job: the atomic
    # WHERE state=<expected> guard means the loser gets ConcurrentModification.
    s = JobStore(tmp_path / "j.db")
    j = s.create("rollout")
    s.transition(j.id, S.RUNNING)                   # "worker 1" claimed it (DB=RUNNING)
    # "worker 2" acted on a stale PENDING snapshot:
    s.get = lambda _id: Job(id=j.id, kind="rollout", state=S.PENDING)
    with pytest.raises(ConcurrentModification):
        s.transition(j.id, S.RUNNING)               # UPDATE WHERE state=PENDING -> 0 rows


def test_missing_job_raises(tmp_path):
    s = JobStore(tmp_path / "j.db")
    with pytest.raises(JobNotFound):
        s.get("nope")
