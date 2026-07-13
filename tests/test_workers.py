"""Pillar 2 tests: claiming, leases, retry/dead-letter, idempotency,
worker-death recovery, and a real multi-process pool drain."""
import pytest

from shoprl.platform import JobState, JobStore, Worker, run_local_pool
from shoprl.platform.store import ConcurrentModification

S = JobState


# --- claim / lease ---------------------------------------------------------
def test_claim_moves_pending_to_running_with_lease(tmp_path):
    s = JobStore(tmp_path / "j.db")
    j = s.create("echo", {"x": 1})
    claimed = s.claim(lease_seconds=30.0, now=1000.0)
    assert claimed.id == j.id and claimed.state is S.RUNNING
    assert claimed.lease_expires_at == pytest.approx(1030.0)


def test_claim_returns_none_when_empty(tmp_path):
    s = JobStore(tmp_path / "j.db")
    assert s.claim() is None


def test_claim_is_single_ownership(tmp_path):
    # One PENDING job: the first claim takes it, the second finds nothing.
    s = JobStore(tmp_path / "j.db")
    s.create("echo")
    assert s.claim() is not None
    assert s.claim() is None


def test_claim_respects_kind_filter(tmp_path):
    s = JobStore(tmp_path / "j.db")
    s.create("rollout"); s.create("optimize")
    got = s.claim(kinds=["optimize"])
    assert got.kind == "optimize"


# --- worker happy path + idempotency --------------------------------------
def test_worker_success_records_result_and_completes(tmp_path):
    s = JobStore(tmp_path / "j.db")
    j = s.create("echo", {"a": 2})
    w = Worker(s, {"echo": lambda job: {"seen": job.payload}})
    out = w.run_once()
    assert out.id == j.id and out.state is S.SUCCEEDED
    assert s.get_result(j.id) == {"seen": {"a": 2}}


def test_record_result_is_idempotent(tmp_path):
    s = JobStore(tmp_path / "j.db")
    j = s.create("echo")
    assert s.record_result(j.id, {"v": 1}) is True     # newly recorded
    assert s.record_result(j.id, {"v": 999}) is False  # duplicate ignored
    assert s.get_result(j.id) == {"v": 1}              # first write wins


def test_idempotent_skip_when_result_already_exists(tmp_path):
    # Simulates redelivery after a crash that occurred *after* the result was
    # recorded: the handler must NOT run again; the job just completes.
    s = JobStore(tmp_path / "j.db")
    j = s.create("echo")
    s.record_result(j.id, {"already": "done"})

    def boom(job):
        raise AssertionError("handler must not run for an already-resulted job")

    out = Worker(s, {"echo": boom}).run_once()
    assert out.state is S.SUCCEEDED
    assert s.get_result(j.id) == {"already": "done"}


# --- failure -> bounded retry -> dead letter -------------------------------
def test_worker_failure_requeues_then_dead_letters(tmp_path):
    s = JobStore(tmp_path / "j.db")
    j = s.create("echo", max_attempts=2)

    def always_fail(job):
        raise ValueError("nope")

    w = Worker(s, {"echo": always_fail})
    first = w.run_once()                    # attempt 1 fails -> requeued PENDING
    assert first.state is S.PENDING and first.attempts == 1
    second = w.run_once()                   # attempt 2 fails -> exhausted
    assert second.state is S.DEAD_LETTER and second.attempts == 2


def test_missing_handler_fails_the_job(tmp_path):
    s = JobStore(tmp_path / "j.db")
    s.create("mystery", max_attempts=1)
    out = Worker(s, {}).run_once()
    assert out.state is S.DEAD_LETTER and "no handler" in out.error


# --- worker-death recovery (lease reaper) ----------------------------------
def test_reap_expired_requeues_dead_worker_job(tmp_path):
    # A worker claims with a short lease, then "dies" (never completes/renews).
    # After the lease passes, the reaper requeues the job for another worker.
    s = JobStore(tmp_path / "j.db")
    j = s.create("echo", max_attempts=3)
    s.claim(lease_seconds=5.0, now=100.0)               # claimed, lease -> 105
    assert s.get(j.id).state is S.RUNNING
    reaped = s.reap_expired(now=200.0)                  # lease long expired
    assert [r.id for r in reaped] == [j.id]
    back = s.get(j.id)
    assert back.state is S.PENDING and back.attempts == 1  # counted as a failure
    assert s.claim() is not None                        # claimable again


def test_reap_dead_letters_when_attempts_exhausted(tmp_path):
    s = JobStore(tmp_path / "j.db")
    j = s.create("echo", max_attempts=1)
    s.claim(lease_seconds=1.0, now=0.0)
    reaped = s.reap_expired(now=100.0)
    assert reaped[0].state is S.DEAD_LETTER


def test_renew_lease_prevents_reap(tmp_path):
    s = JobStore(tmp_path / "j.db")
    j = s.create("echo")
    s.claim(lease_seconds=5.0, now=100.0)               # lease -> 105
    s.renew_lease(j.id, lease_seconds=5.0, now=104.0)   # healthy heartbeat -> 109
    assert s.reap_expired(now=106.0) == []              # not expired -> not reaped
    assert s.get(j.id).state is S.RUNNING


def test_renew_on_non_running_raises(tmp_path):
    s = JobStore(tmp_path / "j.db")
    j = s.create("echo")                                # still PENDING
    with pytest.raises(ConcurrentModification):
        s.renew_lease(j.id)


# --- integration: a real local-process pool drains the queue ---------------
def test_local_pool_drains_queue_across_processes(tmp_path):
    # Seed N jobs, then spawn 3 LOCAL worker PROCESSES that drain and exit.
    # Proves concurrency-safe claiming across real OS processes on one machine.
    path = str(tmp_path / "pool.db")
    s = JobStore(path)
    n = 24
    ids = [s.create("echo", {"i": i}).id for i in range(n)]
    s.close()

    run_local_pool(path, n_workers=3, handlers={"echo": "echo"})

    s2 = JobStore(path)
    assert s2.counts() == {"succeeded": n}              # all done, none lost/stuck
    assert all(s2.get_result(i) is not None for i in ids)  # each produced a result
