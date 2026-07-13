"""Resource scheduler + admission control tests."""
from shoprl.platform import JobStore, ResourceConfig, Scheduler
from shoprl.platform.jobs import JobState
from shoprl.platform.scheduler import CPU, GPU

S = JobState


def _sched(tmp_path, **cfg):
    store = JobStore(tmp_path / "j.db")
    return Scheduler(store, ResourceConfig(**cfg)), store


# --- which job gets the GPU ------------------------------------------------
def test_highest_priority_gpu_job_is_admitted_first(tmp_path):
    sch, store = _sched(tmp_path, gpu_slots=1, max_concurrent_jobs=8)
    lo = sch.submit("optimize", resource=GPU, priority=1)
    hi = sch.submit("optimize", resource=GPU, priority=5)
    admitted = sch.schedule(now=0.0)
    assert [j.id for j in admitted] == [hi.id]        # priority 5 wins the 1 slot
    assert store.get(hi.id).state is S.RUNNING
    assert store.get(lo.id).state is S.PENDING        # still queued


def test_two_gpu_jobs_one_slot_second_waits(tmp_path):
    sch, store = _sched(tmp_path, gpu_slots=1)
    a = sch.submit("optimize", resource=GPU, priority=0)
    b = sch.submit("optimize", resource=GPU, priority=0)   # same priority
    admitted = sch.schedule(now=0.0)
    assert len(admitted) == 1 and admitted[0].id == a.id   # ties -> oldest first
    assert store.get(b.id).state is S.PENDING


# --- capacity full ---------------------------------------------------------
def test_nothing_admitted_when_gpu_full(tmp_path):
    sch, store = _sched(tmp_path, gpu_slots=1)
    sch.submit("optimize", resource=GPU)
    sch.schedule(now=0.0)                              # fills the 1 gpu slot
    sch.submit("optimize", resource=GPU)              # a second arrives
    assert sch.schedule(now=0.0) == []                # capacity full -> queued
    assert sch.status()["gpu"] == {"used": 1, "slots": 1, "queued": 1}


def test_global_max_concurrent_caps_admission(tmp_path):
    sch, store = _sched(tmp_path, gpu_slots=4, cpu_worker_slots=4,
                        max_concurrent_jobs=2)
    for _ in range(4):
        sch.submit("rollout", resource=CPU)
    admitted = sch.schedule(now=0.0)
    assert len(admitted) == 2                          # global cap, not slot cap
    assert sch.status()["at_capacity"] is True


# --- cpu not blocked behind a starved gpu job (backfill) -------------------
def test_cpu_jobs_not_blocked_by_full_gpu(tmp_path):
    sch, store = _sched(tmp_path, gpu_slots=1, cpu_worker_slots=2)
    sch.submit("optimize", resource=GPU, priority=9)
    sch.schedule(now=0.0)                              # gpu full
    c1 = sch.submit("rollout", resource=CPU, priority=0)
    c2 = sch.submit("rollout", resource=CPU, priority=0)
    admitted = sch.schedule(now=0.0)
    assert {j.id for j in admitted} == {c1.id, c2.id}  # cpu backfills despite gpu full


# --- resource release ------------------------------------------------------
def test_completion_releases_slot_for_next_job(tmp_path):
    sch, store = _sched(tmp_path, gpu_slots=1)
    a = sch.submit("optimize", resource=GPU)
    b = sch.submit("optimize", resource=GPU)
    sch.schedule(now=0.0)                              # a runs, b waits
    assert store.get(b.id).state is S.PENDING
    sch.complete(a.id)                                 # frees the gpu slot
    admitted = sch.schedule(now=0.0)
    assert [j.id for j in admitted] == [b.id]          # b now admitted
    assert store.get(b.id).state is S.RUNNING


def test_cancel_pending_dequeues_and_cancel_running_releases(tmp_path):
    sch, store = _sched(tmp_path, gpu_slots=1)
    running = sch.submit("optimize", resource=GPU)
    queued = sch.submit("optimize", resource=GPU)
    sch.schedule(now=0.0)
    # cancel the queued one -> dequeued, never ran
    assert sch.cancel(queued.id).state is S.CANCELLED
    # cancel the running one -> releases the gpu slot
    assert sch.cancel(running.id).state is S.CANCELLED
    assert sch.status()["gpu"]["used"] == 0
    # a newly submitted job can now be admitted
    n = sch.submit("optimize", resource=GPU)
    assert [j.id for j in sch.schedule(now=0.0)] == [n.id]


# --- status ----------------------------------------------------------------
def test_status_reports_usage_and_queue(tmp_path):
    sch, store = _sched(tmp_path, gpu_slots=1, cpu_worker_slots=2)
    sch.submit("optimize", resource=GPU)
    sch.submit("rollout", resource=CPU)
    sch.submit("rollout", resource=CPU)
    sch.submit("rollout", resource=CPU)               # one cpu will queue
    sch.schedule(now=0.0)
    st = sch.status()
    assert st["gpu"]["used"] == 1 and st["gpu"]["queued"] == 0
    assert st["cpu"]["used"] == 2 and st["cpu"]["queued"] == 1
