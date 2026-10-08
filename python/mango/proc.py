"""Process containers, locks and polled subprocesses for the pipeline (docs/DESIGN.md
section 6.5.1, "Process lifecycle").

- Windows: a run-level Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, and one nested
  Job Object per background job. Every process the pipeline launches is assigned to the
  run-level object (and a supervisor also to its job's object); the children a process
  spawns inherit the membership, so when the pipeline dies in any way the OS closes the
  handle and terminates the whole tree, and one job's tree can be stopped and waited for
  on its own.
- POSIX: no container; a supervisor leads its own process group and watches its parent
  (mango/bgjob.py). Job locks are `flock` locks on a descriptor inherited by the
  supervisor and every child it starts, so a lock stays held while any process of the
  job is alive.
- File locks: `.pipeline.lock` (one pipeline per run) and the POSIX job locks.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

WINDOWS = sys.platform == "win32"

if WINDOWS:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                       "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _BasicLimit(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class _ExtendedLimit(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BasicLimit), ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class _BasicAccounting(ctypes.Structure):
        _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                    ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                    ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                    ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD)]

    _k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    _k32.SetInformationJobObject.restype = wintypes.BOOL
    _k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
                                               ctypes.POINTER(wintypes.DWORD)]
    _k32.QueryInformationJobObject.restype = wintypes.BOOL
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.AssignProcessToJobObject.restype = wintypes.BOOL
    _k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _k32.TerminateJobObject.restype = wintypes.BOOL
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.CloseHandle.restype = wintypes.BOOL
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    _k32.GetExitCodeProcess.restype = wintypes.BOOL

    _JOB_BASIC_ACCOUNTING = 1
    _JOB_EXTENDED_LIMIT = 9
    _KILL_ON_JOB_CLOSE = 0x2000
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _STILL_ACTIVE = 259
    _ERROR_ACCESS_DENIED = 5
else:
    import fcntl
    import signal


def _win_error(what: str) -> OSError:
    err = ctypes.get_last_error()  # type: ignore[name-defined]
    return OSError(err, f"{what} failed: {ctypes.FormatError(err).strip()}")  # type: ignore[name-defined]


class JobObject:
    """A Windows Job Object that kills its processes when its last handle closes."""

    def __init__(self) -> None:
        if not WINDOWS:
            raise OSError("Job Objects exist on Windows only")
        h = _k32.CreateJobObjectW(None, None)
        if not h:
            raise _win_error("CreateJobObject")
        self.handle: int | None = h
        info = _ExtendedLimit()
        info.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
        if not _k32.SetInformationJobObject(h, _JOB_EXTENDED_LIMIT, ctypes.byref(info), ctypes.sizeof(info)):
            err = _win_error("SetInformationJobObject")
            self.close()
            raise err

    def assign(self, proc: subprocess.Popen) -> None:
        if not _k32.AssignProcessToJobObject(self.handle, int(proc._handle)):  # type: ignore[attr-defined]
            raise _win_error("AssignProcessToJobObject")

    def active_processes(self) -> int:
        info = _BasicAccounting()
        if not _k32.QueryInformationJobObject(self.handle, _JOB_BASIC_ACCOUNTING, ctypes.byref(info),
                                              ctypes.sizeof(info), None):
            raise _win_error("QueryInformationJobObject")
        return int(info.ActiveProcesses)

    def terminate(self, exit_code: int = 1) -> None:
        if not _k32.TerminateJobObject(self.handle, exit_code):
            raise _win_error("TerminateJobObject")

    def wait_empty(self, timeout: float = 30.0) -> bool:
        deadline = time.time() + timeout
        while self.active_processes() > 0:
            if time.time() > deadline:
                return False
            time.sleep(0.05)
        return True

    def close(self) -> None:
        if self.handle:
            _k32.CloseHandle(self.handle)
            self.handle = None

    def __del__(self) -> None:
        self.close()


class RunContainer:
    """Every process of a run (DESIGN 6.5.1, item 1). On Windows the run-level Job Object
    and nested per-job objects; elsewhere a no-op (supervisors watch their parent)."""

    def __init__(self) -> None:
        self.job = JobObject() if WINDOWS else None

    def new_job(self) -> JobObject | None:
        return JobObject() if WINDOWS else None

    def adopt(self, proc: subprocess.Popen, job: JobObject | None = None) -> None:
        """Assigns a started process to the run-level object and then to `job`, which
        thereby becomes nested inside the run-level object."""
        if self.job is None:
            return
        self.job.assign(proc)
        if job is not None:
            job.assign(proc)

    def close(self) -> None:
        if self.job is not None:
            self.job.close()


def pid_alive(pid: int) -> bool:
    if WINDOWS:
        h = _k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
        try:
            code = wintypes.DWORD()
            if not _k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return True
            return code.value == _STILL_ACTIVE
        finally:
            _k32.CloseHandle(h)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class FileLock:
    """An exclusive, non-blocking OS lock on a file (released when the file is closed or
    the process dies; on POSIX when every process holding the open file description has
    exited)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.fh: Any = None

    def try_acquire(self) -> bool:
        if self.fh is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")
        try:
            if WINDOWS:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self.fh = fh
        return True

    def fileno(self) -> int:
        return self.fh.fileno()

    def release(self) -> None:
        if self.fh is not None:
            fh, self.fh = self.fh, None
            if WINDOWS:
                try:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            fh.close()


# One pipeline per run and process: Pipeline objects created again in the same process
# (restart tests) share the run lock; another process is refused.
_RUN_LOCKS: dict[str, list[Any]] = {}


class RunLockError(RuntimeError):
    pass


def _acquire_shared(path: str | Path) -> str | None:
    """A lock shared by the objects of this process (reference-counted); None if another
    process holds it."""
    key = str(Path(path).resolve())
    held = _RUN_LOCKS.get(key)
    if held is not None:
        held[1] += 1
        return key
    lock = FileLock(path)
    if not lock.try_acquire():
        return None
    _RUN_LOCKS[key] = [lock, 1]
    return key


def acquire_run_lock(path: str | Path) -> str:
    key = _acquire_shared(path)
    if key is None:
        raise RunLockError(f"{path} is held by another pipeline process on this run")
    return key


def release_run_lock(key: str) -> None:
    held = _RUN_LOCKS.get(key)
    if held is None:
        return
    held[1] -= 1
    if held[1] <= 0:
        held[0].release()
        del _RUN_LOCKS[key]


def child_env() -> dict[str, str]:
    """The environment of a supervisor: python/ on the path, so `-m mango.bgjob` resolves."""
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def run_polled(args: list[Any], stderr_path: str | Path, poll: Callable[[], None] | None = None,
               started: Callable[[subprocess.Popen], None] | None = None, pass_fds: tuple[int, ...] = (),
               interval: float = 0.1, popen_kwargs: dict[str, Any] | None = None) -> tuple[int, str]:
    """Runs a command with stdout captured by a reader thread and stderr appended to a log,
    calling `poll` while it waits (the background-job hook). Whatever interrupts the wait
    (an exception from `poll` or `started`, KeyboardInterrupt) terminates the process and
    waits for it. Returns (exit code, stdout)."""
    out: list[str] = []
    # `pass_fds` from the argument and from `popen_kwargs` (ForegroundGuard.wrap on POSIX) are merged.
    kwargs = dict(popen_kwargs or {})
    fds = tuple(pass_fds) + tuple(fd for fd in kwargs.pop("pass_fds", ()) if fd not in pass_fds)
    with open(stderr_path, "a", encoding="utf-8") as errlog:
        proc = subprocess.Popen([str(a) for a in args], stdout=subprocess.PIPE, stderr=errlog, text=True,
                                pass_fds=fds, close_fds=True, **kwargs)
        reader = threading.Thread(target=lambda: out.append(proc.stdout.read()), daemon=True)  # type: ignore[union-attr]
        try:
            if started is not None:
                started(proc)
            reader.start()
            while proc.poll() is None:
                if poll is not None:
                    poll()
                time.sleep(interval)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            if reader.is_alive() or reader.ident is not None:
                reader.join()
            elif proc.stdout is not None:
                proc.stdout.close()
    return proc.returncode, "".join(out)


def terminate_group(pgid: int, sig: int | None = None) -> None:
    """POSIX: signals a process group (no error if it is gone)."""
    if WINDOWS:
        return
    try:
        os.killpg(int(pgid), signal.SIGTERM if sig is None else sig)
    except (ProcessLookupError, PermissionError):
        pass


class ForegroundGuard:
    """Parent-death protection of the pipeline's own subprocesses (self-play, synchronous
    gate and ladder matches; DESIGN 6.5.1). Windows: nothing to do, they are in the
    run-level kill-on-close Job Object. POSIX: every such command runs under
    `python -m mango.bgjob --exec` (its own process group, watching the pipeline, killing
    and reaping its child when the pipeline dies) and inherits the descriptor of
    jobs/foreground.lock, which the pipeline holds for its life. A restarted pipeline takes
    that lock before it launches anything; while an orphan of the previous one is alive it
    signals the recorded process groups, waits up to 60 s and otherwise fails, so a phase is
    never dispatched twice at the same time."""

    def __init__(self, run_dir: str | Path, log: Callable[[str], None] | None = None):
        self.key: str | None = None
        if WINDOWS:
            return
        self.path = Path(run_dir) / "jobs" / "foreground.lock"
        self.pgids = Path(run_dir) / "jobs" / "foreground.pgids"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fresh = str(self.path.resolve()) not in _RUN_LOCKS
        key = _acquire_shared(self.path)
        if key is None:
            groups = self._recorded()
            if log:
                log(f"foreground processes of a previous pipeline are alive (process groups {groups}); signalling them")
            for g in groups:
                terminate_group(g)
            deadline = time.time() + 60.0
            while key is None and time.time() < deadline:
                time.sleep(0.2)
                key = _acquire_shared(self.path)
            if key is None:
                raise RuntimeError(f"{self.path} is still held 60 s after signalling process groups {groups}; "
                                   f"stop those processes and restart")
        self.key = key
        if fresh:
            self.pgids.write_text("", encoding="utf-8")  # every earlier holder has exited

    def _recorded(self) -> list[int]:
        if not self.pgids.exists():
            return []
        return [int(x) for x in self.pgids.read_text(encoding="utf-8").split() if x.strip()]

    def wrap(self, args: list[Any]) -> tuple[list[Any], dict[str, Any]]:
        """The command line and Popen options of a supervised foreground command."""
        if WINDOWS or self.key is None:
            return list(args), {}
        fd = _RUN_LOCKS[self.key][0].fileno()
        wrapped = [sys.executable, "-m", "mango.bgjob", "--exec", "--parent-pid", str(os.getpid()), "--lock-fd", str(fd),
                   "--"] + [str(a) for a in args]
        return wrapped, {"pass_fds": (fd,), "start_new_session": True, "env": child_env()}

    def record(self, proc: subprocess.Popen) -> None:
        if not WINDOWS and self.key is not None:
            with open(self.pgids, "a", encoding="utf-8") as f:
                f.write(f"{proc.pid}\n")

    def release(self) -> None:
        if self.key is not None:
            release_run_lock(self.key)
            self.key = None
