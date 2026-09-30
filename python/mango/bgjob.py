"""Background-job supervisor (docs/DESIGN.md section 6.5.1).

    python -m mango.bgjob --spec runs/<name>/jobs/<job id>.json --parent-pid P [--lock-fd N]
    python -m mango.bgjob --exec --parent-pid P [--lock-fd N] -- COMMAND ...

--exec (the pipeline's foreground commands on POSIX, proc.ForegroundGuard): runs COMMAND
with stdout and stderr inherited and exits with its code; on the parent's death or SIGTERM
it terminates the command, reaps its group and exits 3. No `go` handshake (there is no
container to join first); the lock descriptor is passed on to the command.

Starts no work until it reads the line `go` on stdin (the pipeline writes it after the
supervisor was placed in its Job Object); on EOF or any other input it exits with code 2
without starting a child. Then it runs the job described by the spec:

- kind "gate": the gate's `mango_match` command, unchanged (report matches/<i>.json);
- kind "ladder": the strength step for one model (add the initial model if the ladder has
  no model entry, add the model, play its missing matches, publish the fit of its
  iteration), every `mango_match` a polled child.

POSIX: the supervisor leads its own process group; once a second it checks that its
parent pipeline is alive (os.getppid() against --parent-pid). On the parent's death or
SIGTERM it terminates its child, signals its group, waits for every child and exits (3).
The job lock descriptor (--lock-fd) is passed on to every child, so the lock is held
until the last process of the job has exited. Exit codes: 0 done, 1 failed, 2 no `go`,
3 aborted.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from .proc import WINDOWS, run_polled
from .state import read_json


class Aborted(Exception):
    pass


class Supervisor:
    def __init__(self, run: Path | None, parent_pid: int, lock_fd: int, log_name: str | None):
        self.run = run
        self.parent_pid = parent_pid
        self.lock_fd = lock_fd
        self.log_path = run / "logs" / log_name if run is not None and log_name else None
        self.abort = threading.Event()

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} [bgjob {os.getpid()}] {msg}\n"
        if self.log_path is None:
            sys.stderr.write(line)
            sys.stderr.flush()
            return
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)

    def _poll(self) -> None:
        if self.abort.is_set():
            raise Aborted()

    def runner(self, args: list[Any], log_name: str) -> str:
        """The ladder's runner: one polled child per command, stderr to the job's log."""
        self._poll()
        self.log("exec " + " ".join(str(a) for a in args))
        fds = (self.lock_fd,) if self.lock_fd >= 0 else ()
        rc, out = run_polled(args, self.log_path, poll=self._poll, pass_fds=fds)
        if rc != 0:
            raise RuntimeError(f"{args[0]} failed with exit code {rc}")
        return out

    def watch_parent(self) -> None:
        """POSIX: the parent's death or SIGTERM aborts the job; after 10 s without a clean
        exit the whole group is killed (this process included)."""
        import signal

        signal.signal(signal.SIGTERM, lambda *_: self.abort.set())

        def watch() -> None:
            while not self.abort.is_set():
                if os.getppid() != self.parent_pid:
                    self.log("parent pipeline died; aborting")
                    self.abort.set()
                    break
                time.sleep(1.0)
            time.sleep(10.0)
            os.killpg(os.getpgid(0), signal.SIGKILL)

        threading.Thread(target=watch, daemon=True).start()

    def reap_group(self) -> None:
        """POSIX, on abort: signal the group (children that are not ours to poll, e.g.
        GTP engines) and wait for every child to exit before this process exits."""
        import signal

        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            os.killpg(os.getpgid(0), signal.SIGTERM)
        except OSError:
            pass
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == 0:
                time.sleep(0.05)


def run_gate(spec: dict[str, Any], sup: Supervisor) -> None:
    out = sup.runner(spec["command"], "match_bg.log")
    if out.strip():
        sup.log(out.strip().splitlines()[-1])


def run_ladder(spec: dict[str, Any], sup: Supervisor) -> None:
    from .config import load_config
    from .strength import Ladder, ladder_step

    cfg = load_config(sup.run / "config.json")
    ladder = Ladder(sup.run, cfg, spec["bin"], int(spec["run_seed"]), spec["device"], runner=sup.runner, log=sup.log,
                    match_command=spec["match_command"])
    t0 = time.time()
    fit = ladder_step(ladder, spec["initial_model_id"], spec["model_id"], int(spec["iteration"]), bool(spec["promoted"]))
    r = fit["ratings"][spec["model_id"]]
    sup.log(f"ladder job {spec['iteration']}: {spec['model_id']} rated {r['elo']:.0f} in {time.time() - t0:.0f}s")


def run_exec(command: list[str], parent_pid: int, lock_fd: int) -> int:
    import subprocess

    sup = Supervisor(None, parent_pid, lock_fd, None)
    if not WINDOWS:
        sup.watch_parent()
    proc = subprocess.Popen(command, pass_fds=(lock_fd,) if lock_fd >= 0 else (), close_fds=True)
    try:
        while proc.poll() is None:
            if sup.abort.is_set():
                raise Aborted()
            time.sleep(0.1)
    except (Aborted, KeyboardInterrupt):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if not WINDOWS:
            sup.reap_group()
        return 3
    rc = proc.returncode
    return rc if rc >= 0 else 128 - rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mango background-job supervisor (DESIGN 6.5.1)")
    ap.add_argument("--spec")
    ap.add_argument("--exec", dest="exec_", action="store_true")
    ap.add_argument("--parent-pid", type=int, required=True)
    ap.add_argument("--lock-fd", type=int, default=-1)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    if args.exec_:
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        return run_exec(command, args.parent_pid, args.lock_fd)
    if not args.spec:
        ap.error("--spec is required without --exec")
    line = sys.stdin.readline()
    if line.strip() != "go":
        return 2
    spec = read_json(args.spec)
    run = Path(spec["run"])
    sup = Supervisor(run, args.parent_pid, args.lock_fd, spec["log"])
    if not WINDOWS:
        sup.watch_parent()
    sup.log(f"job {spec['job_id']} started")
    try:
        if spec["kind"] == "gate":
            run_gate(spec, sup)
        elif spec["kind"] == "ladder":
            run_ladder(spec, sup)
        else:
            raise ValueError(f"unknown job kind {spec['kind']!r}")
    except Aborted:
        sup.log(f"job {spec['job_id']} aborted")
        if not WINDOWS:
            sup.reap_group()
        return 3
    except Exception:
        sup.log(f"job {spec['job_id']} failed:\n{traceback.format_exc()}")
        if not WINDOWS:
            sup.reap_group()
        return 1
    sup.log(f"job {spec['job_id']} finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
