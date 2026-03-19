#!/usr/bin/env python3
"""DGX Spark inference tuning validator.

Checks and optionally applies optimizations for NVIDIA DGX Spark systems
with unified CPU/GPU memory. Designed for inference workloads.

Usage:
    sudo python3 dgx-spark-tune.py             # check + prompt to fix
    sudo python3 dgx-spark-tune.py --check      # check only, no changes
    sudo python3 dgx-spark-tune.py --apply       # apply all without prompting
    sudo python3 dgx-spark-tune.py --autotune    # benchmark GPU clocks and lock optimal
    sudo python3 dgx-spark-tune.py --revert      # undo everything, restore stock
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
NO_COLOR: bool = not sys.stdout.isatty() or "NO_COLOR" in os.environ


def _c(code: str, text: str) -> str:
    return text if NO_COLOR else f"\033[{code}m{text}\033[0m"


def green(t: str) -> str:
    return _c("32", t)


def yellow(t: str) -> str:
    return _c("33", t)


def red(t: str) -> str:
    return _c("31", t)


def bold(t: str) -> str:
    return _c("1", t)


def dim(t: str) -> str:
    return _c("2", t)


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------
def run(cmd: str, *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        check=check,
    )


def sysctl_get(key: str) -> str | None:
    r = run(f"sysctl -n {key}")
    return r.stdout.strip() if r.returncode == 0 else None


def sysctl_set(key: str, value: str) -> bool:
    r = run(f"sysctl -w {key}={value}")
    return r.returncode == 0


def ask(prompt: str) -> bool:
    try:
        return input(f"  {prompt} [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


# ---------------------------------------------------------------------------
# Check result
# ---------------------------------------------------------------------------
@dataclass
class Result:
    name: str
    ok: bool
    current: str
    expected: str
    fix: Any = None  # callable or None

    def display(self) -> None:
        tag = green(" OK ") if self.ok else red("FAIL")
        cur = dim(self.current) if self.ok else yellow(self.current)
        line = f"  [{tag}] {self.name:<50s} {cur}"
        if not self.ok:
            line += f"  (want: {self.expected})"
        print(line)


# ---------------------------------------------------------------------------
# Sysctl checks
# ---------------------------------------------------------------------------
SYSCTL_CONF = "/etc/sysctl.d/90-inference.conf"

SYSCTLS: dict[str, str] = {
    # Network: BBR + fq
    "net.core.default_qdisc": "fq",
    "net.ipv4.tcp_congestion_control": "bbr",
    # Socket buffers
    "net.core.rmem_max": "16777216",
    "net.core.wmem_max": "16777216",
    "net.core.rmem_default": "1048576",
    "net.core.wmem_default": "1048576",
    "net.ipv4.tcp_rmem": "4096\t1048576\t16777216",
    "net.ipv4.tcp_wmem": "4096\t1048576\t16777216",
    # Connection handling
    "net.core.netdev_max_backlog": "16384",
    "net.ipv4.tcp_fastopen": "3",
    "net.ipv4.tcp_slow_start_after_idle": "0",
    "net.ipv4.tcp_mtu_probing": "1",
    "net.ipv4.tcp_tw_reuse": "1",
    # VM: inference workload
    "vm.swappiness": "10",
    "vm.dirty_ratio": "10",
    "vm.dirty_background_ratio": "5",
    # Unified memory: minimize cache for GPU headroom
    "vm.vfs_cache_pressure": "500",
    "vm.dirty_expire_centisecs": "100",
    "vm.dirty_writeback_centisecs": "100",
    "vm.min_free_kbytes": "524288",
    "vm.watermark_scale_factor": "50",
    "vm.page-cluster": "0",
    "vm.zone_reclaim_mode": "1",
}

# These are checked live but persisted in the conf file with different format
SYSCTL_CONF_MAP: dict[str, str] = {
    "net.ipv4.tcp_rmem": "4096 1048576 16777216",
    "net.ipv4.tcp_wmem": "4096 1048576 16777216",
}


def write_sysctl_conf() -> None:
    """Write the full sysctl config file and reload."""
    lines = [
        "# DGX Spark inference tuning — managed by dgx-spark-tune.py",
        "",
        "# --- Network: BBR + fq ---",
        "net.core.default_qdisc = fq",
        "net.ipv4.tcp_congestion_control = bbr",
        "",
        "# --- Network: socket buffers ---",
        "net.core.rmem_max = 16777216",
        "net.core.wmem_max = 16777216",
        "net.core.rmem_default = 1048576",
        "net.core.wmem_default = 1048576",
        "net.ipv4.tcp_rmem = 4096 1048576 16777216",
        "net.ipv4.tcp_wmem = 4096 1048576 16777216",
        "",
        "# --- Network: connection handling ---",
        "net.core.netdev_max_backlog = 16384",
        "net.ipv4.tcp_fastopen = 3",
        "net.ipv4.tcp_slow_start_after_idle = 0",
        "net.ipv4.tcp_mtu_probing = 1",
        "net.ipv4.tcp_tw_reuse = 1",
        "",
        "# --- VM: inference workload ---",
        "vm.swappiness = 10",
        "vm.dirty_ratio = 10",
        "vm.dirty_background_ratio = 5",
        "",
        "# --- Unified memory: minimize cache for GPU headroom ---",
        "vm.vfs_cache_pressure = 500",
        "vm.dirty_expire_centisecs = 100",
        "vm.dirty_writeback_centisecs = 100",
        "vm.min_free_kbytes = 524288",
        "vm.watermark_scale_factor = 50",
        "vm.page-cluster = 0",
        "vm.zone_reclaim_mode = 1",
    ]
    # Preserve existing TFO key if present
    existing = Path(SYSCTL_CONF)
    tfo_key = None
    if existing.exists():
        for line in existing.read_text().splitlines():
            if "tcp_fastopen_key" in line and "=" in line:
                tfo_key = line.strip()
                break
    if tfo_key:
        lines.extend(["", "# --- TFO server key (stable across reboots) ---", tfo_key])
    else:
        # Generate a new key
        r = run("openssl rand -hex 16")
        if r.returncode == 0:
            h = r.stdout.strip()
            key = f"{h[0:8]}-{h[8:16]}-{h[16:24]}-{h[24:32]}"
            lines.extend(
                [
                    "",
                    "# --- TFO server key (stable across reboots) ---",
                    f"net.ipv4.tcp_fastopen_key = {key}",
                ]
            )

    Path(SYSCTL_CONF).write_text("\n".join(lines) + "\n")
    r = run(f"sysctl -p {SYSCTL_CONF}")
    if r.returncode != 0:
        print(yellow(f"  Warning: sysctl reload had errors:\n    {r.stderr.strip()}"))


def check_sysctls() -> list[Result]:
    results = []
    for key, expected in SYSCTLS.items():
        current = sysctl_get(key) or "N/A"
        ok = current == expected
        results.append(
            Result(
                name=f"sysctl {key}",
                ok=ok,
                current=current,
                expected=expected,
            )
        )
    return results


# ---------------------------------------------------------------------------
# TFO key check (just verify one exists and is persisted)
# ---------------------------------------------------------------------------
def check_tfo_key() -> Result:
    current = run("sysctl -n net.ipv4.tcp_fastopen_key").stdout.strip()
    has_key = bool(current) and current != "00000000-00000000-00000000-00000000"
    persisted = False
    conf = Path(SYSCTL_CONF)
    if conf.exists():
        persisted = "tcp_fastopen_key" in conf.read_text()
    ok = has_key and persisted
    status = "set + persisted" if ok else ("set but not persisted" if has_key else "missing")
    return Result(
        name="sysctl net.ipv4.tcp_fastopen_key",
        ok=ok,
        current=status,
        expected="set + persisted",
    )


# ---------------------------------------------------------------------------
# Docker + NVIDIA runtime
# ---------------------------------------------------------------------------
DOCKER_DAEMON_JSON = "/etc/docker/daemon.json"

EXPECTED_DOCKER_CONFIG = {
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime",
        },
    },
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "50m",
        "max-file": "3",
    },
    "default-ulimits": {
        "memlock": {"Name": "memlock", "Hard": -1, "Soft": -1},
        "nofile": {"Name": "nofile", "Hard": 500000, "Soft": 500000},
    },
}


def check_docker() -> list[Result]:
    results = []

    # Docker installed?
    r = run("docker --version")
    docker_installed = r.returncode == 0
    results.append(
        Result(
            name="docker installed",
            ok=docker_installed,
            current=r.stdout.strip() if docker_installed else "not found",
            expected="installed",
        )
    )
    if not docker_installed:
        return results

    # nvidia-container-toolkit?
    r = run("dpkg -l nvidia-container-toolkit 2>/dev/null | grep '^ii'")
    nct = r.returncode == 0
    results.append(
        Result(
            name="nvidia-container-toolkit",
            ok=nct,
            current="installed" if nct else "missing",
            expected="installed",
        )
    )

    # daemon.json exists and has correct config?
    daemon_path = Path(DOCKER_DAEMON_JSON)
    if daemon_path.exists():
        try:
            current = json.loads(daemon_path.read_text())
        except json.JSONDecodeError:
            current = {}
    else:
        current = {}

    def _check_key(key: str, expected: Any) -> Result:
        actual = current.get(key)
        ok = actual == expected
        return Result(
            name=f"docker daemon.json: {key}",
            ok=ok,
            current=json.dumps(actual, separators=(",", ":")) if actual else "not set",
            expected=json.dumps(expected, separators=(",", ":")),
        )

    results.append(_check_key("default-runtime", "nvidia"))
    results.append(_check_key("runtimes", EXPECTED_DOCKER_CONFIG["runtimes"]))
    results.append(_check_key("log-driver", "json-file"))
    results.append(_check_key("log-opts", EXPECTED_DOCKER_CONFIG["log-opts"]))
    results.append(_check_key("default-ulimits", EXPECTED_DOCKER_CONFIG["default-ulimits"]))

    return results


def fix_docker() -> None:
    # Ensure nvidia-container-toolkit is installed
    r = run("dpkg -l nvidia-container-toolkit 2>/dev/null | grep '^ii'")
    if r.returncode != 0:
        print("  Installing nvidia-container-toolkit...")
        run("apt-get update -qq && apt-get install -y -qq nvidia-container-toolkit", check=True)

    daemon_path = Path(DOCKER_DAEMON_JSON)
    existing: dict[str, Any] = {}
    if daemon_path.exists():
        with contextlib.suppress(json.JSONDecodeError):
            existing = json.loads(daemon_path.read_text())
    merged = {**existing, **EXPECTED_DOCKER_CONFIG}
    daemon_path.write_text(json.dumps(merged, indent=4) + "\n")
    run("systemctl restart docker")
    print("  Docker daemon.json merged and service restarted.")


# ---------------------------------------------------------------------------
# WiFi tuning
# ---------------------------------------------------------------------------
NM_POWERSAVE_CONF = "/etc/NetworkManager/conf.d/wifi-powersave.conf"
MT7925_MODPROBE_CONF = "/etc/modprobe.d/mt7925.conf"
MT7925_UDEV_RULE = "/etc/udev/rules.d/99-mt7925-no-pm.rules"


def _find_wifi_interface() -> str | None:
    r = run("iw dev")
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("Interface "):
            iface = line.split()[1]
            if re.fullmatch(r"[a-zA-Z0-9_-]+", iface):
                return iface
    return None


def _wifi_driver(iface: str) -> str:
    """Return kernel driver name for the given interface."""
    r = run(f"readlink /sys/class/net/{iface}/device/driver")
    return Path(r.stdout.strip()).name if r.returncode == 0 else ""


def _wifi_freq(iface: str) -> int:
    """Return current frequency in MHz, or 0."""
    r = run(f"iw dev {iface} info")
    for line in r.stdout.splitlines():
        m = re.search(r"channel\s+\d+\s+\((\d+)\s+MHz\)", line)
        if m:
            return int(m.group(1))
    return 0


def _wifi_pci_addr(iface: str) -> str:
    """Return PCI bus address for the interface, or empty string."""
    r = run(f"readlink /sys/class/net/{iface}/device")
    if r.returncode != 0:
        return ""
    return Path(r.stdout.strip()).name


def _is_mt7925(iface: str) -> bool:
    return "mt7925" in _wifi_driver(iface)


def check_wifi(iface: str) -> list[Result]:
    results: list[Result] = []

    # Power save
    r = run(f"iw dev {iface} get power_save")
    current = r.stdout.strip().split(":")[-1].strip() if r.returncode == 0 else "unknown"
    results.append(
        Result(
            name=f"wifi power_save ({iface})",
            ok=current.lower() == "off",
            current=current,
            expected="off",
        )
    )

    conf = Path(NM_POWERSAVE_CONF)
    persisted = conf.exists() and "wifi.powersave = 2" in conf.read_text()
    results.append(
        Result(
            name="wifi power_save persisted",
            ok=persisted,
            current="persisted" if persisted else "not persisted",
            expected="persisted",
        )
    )

    # MT7925-specific checks
    if not _is_mt7925(iface):
        return results

    # 6 GHz band — known unstable with mt7925 MLD
    freq = _wifi_freq(iface)
    on_6ghz = freq >= 5925
    results.append(
        Result(
            name="mt7925 not on 6 GHz (unstable with MLD)",
            ok=not on_6ghz,
            current=f"{freq} MHz ({'6 GHz' if on_6ghz else '5 GHz' if freq >= 5000 else '2.4 GHz'})",
            expected="5 GHz or 2.4 GHz",
        )
    )

    # NM connection band pinned (not auto — avoids 6 GHz roaming)
    r = run("nmcli -t -f NAME connection show --active")
    conn_name = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else ""
    if conn_name:
        r = run(f"nmcli -t -f 802-11-wireless.band connection show {shlex.quote(conn_name)}")
        band = r.stdout.strip().split(":")[-1] if r.returncode == 0 else ""
        band_ok = band in ("a", "bg")
        results.append(
            Result(
                name="wifi band pinned (NM)",
                ok=band_ok,
                current=f"band={band or 'auto'}",
                expected="band=a or bg (not auto)",
            )
        )

    # ASPM disable in modprobe.d
    modprobe = Path(MT7925_MODPROBE_CONF)
    aspm_ok = modprobe.exists() and "disable_aspm=1" in modprobe.read_text()
    results.append(
        Result(
            name="mt7925 PCI ASPM disabled (modprobe)",
            ok=aspm_ok,
            current="disabled" if aspm_ok else "not configured",
            expected="disabled",
        )
    )

    # PCI runtime PM udev rule
    udev = Path(MT7925_UDEV_RULE)
    udev_ok = udev.exists() and "power/control" in udev.read_text()
    results.append(
        Result(
            name="mt7925 PCI runtime PM disabled (udev)",
            ok=udev_ok,
            current="configured" if udev_ok else "not configured",
            expected="configured",
        )
    )

    return results


def fix_wifi() -> None:
    iface = _find_wifi_interface()
    if not iface:
        print("  No wifi interface found, skipping.")
        return

    # Power save
    run(f"iw dev {iface} set power_save off")
    conf = Path(NM_POWERSAVE_CONF)
    conf.parent.mkdir(parents=True, exist_ok=True)
    conf.write_text("# Disable wifi power save for low latency\n[connection]\nwifi.powersave = 2\n")
    print(f"  Power save disabled on {iface} and persisted.")

    if not _is_mt7925(iface):
        run("systemctl restart NetworkManager")
        return

    # MT7925 stability fixes
    print("  MT7925 detected — applying stability fixes:")

    # Pin to 5 GHz band
    r = run("nmcli -t -f NAME connection show --active")
    conn_name = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else ""
    if conn_name:
        run(f"nmcli connection modify {shlex.quote(conn_name)} wifi.band bg")
        print("    Pinned connection to 2.4 GHz band (most stable for headless).")

    # ASPM disable
    Path(MT7925_MODPROBE_CONF).write_text("options mt7925e disable_aspm=1\n")
    print("    PCI ASPM disabled via modprobe.d.")

    # PCI runtime PM udev rule
    Path(MT7925_UDEV_RULE).write_text(
        "# Prevent MT7925 WiFi from entering PCI power save\n"
        'ACTION=="add", SUBSYSTEM=="pci", ATTR{vendor}=="0x14c3", ATTR{power/control}="on"\n'
    )
    print("    PCI runtime PM disabled via udev rule.")

    # Apply: reconnect on 5 GHz
    run("systemctl restart NetworkManager")
    time.sleep(3)
    if conn_name:
        run(f"nmcli connection up {shlex.quote(conn_name)} 2>/dev/null")
    print("    NetworkManager restarted.")


# ---------------------------------------------------------------------------
# NVIDIA persistence mode
# ---------------------------------------------------------------------------
def check_nvidia() -> list[Result]:
    results = []
    r = run("nvidia-smi -q")
    if r.returncode != 0:
        results.append(Result("nvidia-smi", False, "not available", "working"))
        return results

    for line in r.stdout.splitlines():
        if "Persistence Mode" in line:
            val = line.split(":")[-1].strip()
            results.append(
                Result(
                    name="nvidia persistence mode",
                    ok=val == "Enabled",
                    current=val,
                    expected="Enabled",
                )
            )
            break
    return results


def fix_nvidia_persistence() -> None:
    run("nvidia-smi -pm 1")
    run("systemctl enable nvidia-persistenced 2>/dev/null")
    run("systemctl start nvidia-persistenced 2>/dev/null")
    print("  NVIDIA persistence mode enabled (persistenced service started).")


# ---------------------------------------------------------------------------
# CPU governor
# ---------------------------------------------------------------------------
def check_cpu_governor() -> list[Result]:
    gov_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if not gov_path.exists():
        return [Result("cpu governor", True, "N/A (no cpufreq)", "N/A")]
    current = gov_path.read_text().strip()
    return [
        Result(
            name="cpu governor",
            ok=current == "performance",
            current=current,
            expected="performance",
        )
    ]


def check_cpu_idle() -> list[Result]:
    """Check that deep CPU idle states (LPI-1+) are disabled."""
    results: list[Result] = []
    cpu0_idle = Path("/sys/devices/system/cpu/cpu0/cpuidle")
    if not cpu0_idle.exists():
        return results
    for state_dir in sorted(cpu0_idle.glob("state[0-9]*")):
        name = (state_dir / "name").read_text().strip() if (state_dir / "name").exists() else "?"
        latency_path = state_dir / "latency"
        latency = int(latency_path.read_text().strip()) if latency_path.exists() else 0
        disable_path = state_dir / "disable"
        if not disable_path.exists():
            continue
        disabled = disable_path.read_text().strip() == "1"
        # LPI-0 / WFI (0us latency) is fine to keep enabled
        if latency == 0:
            results.append(
                Result(
                    name=f"cpu idle {name} ({latency}us)",
                    ok=True,
                    current="enabled (ok, zero latency)",
                    expected="enabled",
                )
            )
        else:
            results.append(
                Result(
                    name=f"cpu idle {name} ({latency}us)",
                    ok=disabled,
                    current="disabled" if disabled else "enabled",
                    expected="disabled",
                )
            )
    return results


def fix_cpu_governor() -> None:
    cpus = Path("/sys/devices/system/cpu/")
    for gov in sorted(cpus.glob("cpu*/cpufreq/scaling_governor")):
        gov.write_text("performance\n")
    print("  All CPUs set to performance governor.")


def fix_cpu_idle() -> None:
    """Disable deep CPU idle states (LPI-1+) on all CPUs and persist via service."""
    count = 0
    cpus = Path("/sys/devices/system/cpu/")
    for state_dir in sorted(cpus.glob("cpu*/cpuidle/state[0-9]*")):
        latency_path = state_dir / "latency"
        latency = int(latency_path.read_text().strip()) if latency_path.exists() else 0
        if latency == 0:
            continue  # keep WFI enabled
        disable_path = state_dir / "disable"
        if disable_path.exists() and disable_path.read_text().strip() != "1":
            disable_path.write_text("1\n")
            count += 1
    print(f"  Disabled {count} deep CPU idle states (kept WFI).")
    _write_inference_tune_service()
    print("  Persisted via inference-tune.service.")


# ---------------------------------------------------------------------------
# Sleep / suspend prevention
# ---------------------------------------------------------------------------
LOGIND_NOSLEEP_CONF = "/etc/systemd/logind.conf.d/nosleep.conf"

SLEEP_TARGETS = ["sleep.target", "suspend.target", "hibernate.target", "hybrid-sleep.target"]

GNOME_POWER_KEYS: dict[str, tuple[str, str]] = {
    # (schema.key): (expected, gsettings set value)
    "org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type": ("'nothing'", "nothing"),
    "org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type": ("'nothing'", "nothing"),
    "org.gnome.settings-daemon.plugins.power lid-close-ac-action": ("'nothing'", "nothing"),
    "org.gnome.settings-daemon.plugins.power lid-close-battery-action": ("'nothing'", "nothing"),
}


def _desktop_users() -> list[str]:
    """Return logged-in GUI users (not root — root has no persistent dconf)."""
    users: list[str] = []
    # SUDO_USER is the user who invoked sudo — most reliable
    sudo_user = os.environ.get("SUDO_USER", "")
    if sudo_user and sudo_user != "root":
        users.append(sudo_user)
    # Also check who is logged in
    r = run("who 2>/dev/null")
    for line in r.stdout.splitlines():
        parts = line.split()
        if parts and parts[0] != "root" and parts[0] not in users:
            users.append(parts[0])
    return users


def _gsettings_cmd(user: str, subcmd: str) -> str:
    """Build a gsettings command that runs as the given user with proper dbus."""
    if user == "root":
        return f"gsettings {subcmd}"
    u = shlex.quote(user)
    return (
        f"sudo -u {u} DBUS_SESSION_BUS_ADDRESS="
        f'"unix:path=/run/user/$(id -u {u})/bus" '
        f"gsettings {subcmd}"
    )


def check_sleep() -> list[Result]:
    results = []

    # systemd sleep targets masked?
    for target in SLEEP_TARGETS:
        r = run(f"systemctl is-enabled {target} 2>/dev/null")
        state = r.stdout.strip()
        ok = state == "masked"
        results.append(
            Result(
                name=f"systemd {target}",
                ok=ok,
                current=state or "unknown",
                expected="masked",
            )
        )

    # logind config
    conf = Path(LOGIND_NOSLEEP_CONF)
    logind_ok = conf.exists() and "IdleAction=ignore" in conf.read_text()
    results.append(
        Result(
            name="logind idle/lid/suspend overrides",
            ok=logind_ok,
            current="configured" if logind_ok else "not configured",
            expected="configured",
        )
    )

    # GNOME settings — check each desktop user
    r = run("which gsettings 2>/dev/null")
    if r.returncode != 0:
        return results

    for user in _desktop_users():
        for schema_key, (expected, _) in GNOME_POWER_KEYS.items():
            r = run(_gsettings_cmd(user, f"get {schema_key}") + " 2>/dev/null")
            current = r.stdout.strip()
            ok = current == expected
            short_name = schema_key.split()[-1]
            results.append(
                Result(
                    name=f"gnome {short_name} ({user})",
                    ok=ok,
                    current=current or "N/A",
                    expected=expected,
                )
            )

        r = run(_gsettings_cmd(user, "get org.gnome.desktop.session idle-delay") + " 2>/dev/null")
        current = r.stdout.strip()
        ok = current == "uint32 0"
        results.append(
            Result(
                name=f"gnome idle-delay ({user})",
                ok=ok,
                current=current or "N/A",
                expected="uint32 0",
            )
        )

    return results


def fix_sleep() -> None:
    # Mask systemd sleep targets
    for target in SLEEP_TARGETS:
        run(f"systemctl mask {target}")
    print("  Masked systemd sleep targets.")

    # logind overrides
    conf = Path(LOGIND_NOSLEEP_CONF)
    conf.parent.mkdir(parents=True, exist_ok=True)
    conf.write_text(
        "[Login]\n"
        "HandleSuspendKey=ignore\n"
        "HandleHibernateKey=ignore\n"
        "HandleLidSwitch=ignore\n"
        "HandleLidSwitchExternalPower=ignore\n"
        "HandleLidSwitchDocked=ignore\n"
        "IdleAction=ignore\n"
        "IdleActionSec=0\n"
    )
    run("systemctl restart systemd-logind")
    print("  logind overrides written and service restarted.")

    # GNOME settings — apply to root + all desktop users
    r = run("which gsettings 2>/dev/null")
    if r.returncode != 0:
        return
    for user in _desktop_users():
        for schema_key, (_, value) in GNOME_POWER_KEYS.items():
            run(_gsettings_cmd(user, f"set {schema_key} '{value}'"))
        run(_gsettings_cmd(user, "set org.gnome.desktop.session idle-delay 0"))
        print(f"  GNOME power/sleep settings disabled for {user}.")


# ---------------------------------------------------------------------------
# CUDA toolkit
# ---------------------------------------------------------------------------


def _parse_version(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v))


def _get_installed_cuda_version() -> str | None:
    """Get the highest installed cuda-toolkit-X-Y version."""
    r = run("dpkg -l 'cuda-toolkit-[0-9]*-[0-9]*' 2>/dev/null | grep '^ii'")
    if r.returncode != 0 or not r.stdout.strip():
        return None
    best: str | None = None
    best_v: tuple[int, ...] = ()
    for line in r.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        pkg = parts[1]  # e.g. cuda-toolkit-13-2
        m = re.match(r"cuda-toolkit-(\d+)-(\d+)", pkg)
        if m:
            v = (int(m.group(1)), int(m.group(2)))
            if v > best_v:
                best_v = v
                best = f"{m.group(1)}.{m.group(2)}"
    return best


def _get_available_cuda_version() -> str | None:
    """Get the latest cuda-toolkit version available in apt repos."""
    r = run("apt-cache madison cuda-toolkit 2>/dev/null")
    if r.returncode != 0 or not r.stdout.strip():
        return None
    best: str | None = None
    best_v: tuple[int, ...] = ()
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        m = re.match(r"(\d+\.\d+)", parts[1])
        if m:
            v = _parse_version(m.group(1))
            if v > best_v:
                best_v = v
                best = m.group(1)
    return best


def _nvcc_version() -> str | None:
    """Get nvcc version from /usr/local/cuda (follows alternatives symlink)."""
    r = run("/usr/local/cuda/bin/nvcc --version")
    if r.returncode != 0:
        return None
    for line in r.stdout.splitlines():
        m = re.search(r"release (\d+\.\d+)", line)
        if m:
            return m.group(1)
    return None


def _cuda_alternative() -> str | None:
    """Check what /usr/local/cuda points to."""
    cuda = Path("/usr/local/cuda")
    if not cuda.is_symlink():
        return None
    target = str(cuda.resolve())
    m = re.search(r"cuda-(\d+\.\d+)", target)
    return m.group(1) if m else None


def check_cuda() -> list[Result]:
    results = []

    installed = _get_installed_cuda_version()
    latest = _get_available_cuda_version()
    latest_v = _parse_version(latest) if latest else ()
    installed_v = _parse_version(installed) if installed else ()

    # Is any CUDA installed?
    results.append(
        Result(
            name="cuda-toolkit installed",
            ok=installed is not None,
            current=installed or "not installed",
            expected="installed",
        )
    )

    # Is it the latest available?
    if installed and latest:
        up_to_date = installed_v >= latest_v
        results.append(
            Result(
                name="cuda-toolkit is latest",
                ok=up_to_date,
                current=installed,
                expected=f"{latest} (repo latest)",
            )
        )

    # Check nvcc works
    nvcc = _nvcc_version()
    results.append(
        Result(
            name="nvcc functional",
            ok=nvcc is not None,
            current=nvcc or "not found",
            expected="working",
        )
    )

    # Check /usr/local/cuda symlink
    alt = _cuda_alternative()
    results.append(
        Result(
            name="/usr/local/cuda symlink",
            ok=alt is not None,
            current=f"cuda-{alt}" if alt else "not set",
            expected="set",
        )
    )

    return results


def fix_cuda() -> None:
    latest = _get_available_cuda_version()
    if not latest:
        print("  Updating apt cache...")
        run("apt-get update -qq")
        latest = _get_available_cuda_version()
    if not latest:
        print(red("  Cannot determine latest CUDA version from repositories."))
        return

    major, minor = latest.split(".")
    pkg = f"cuda-toolkit-{major}-{minor}"

    print(f"  Installing {pkg} (latest)...")
    r = run(f"apt-get install -y {pkg}")
    if r.returncode != 0:
        print(red(f"  Install failed: {r.stderr.strip()[-200:]}"))
        return

    print(green(f"  {pkg} installed successfully."))


# ---------------------------------------------------------------------------
# GPU clock autotune
# ---------------------------------------------------------------------------
AUTOTUNE_CONF = "/etc/nvidia-gpu-clock.conf"
AUTOTUNE_SERVICE = "/etc/systemd/system/nvidia-gpu-clock.service"
INFERENCE_TUNE_SERVICE = "/etc/systemd/system/inference-tune.service"


def _bench_kernel_source() -> str:
    """Generate the CUDA benchmark kernel source code."""
    lines = [
        "#include <cstdio>",
        "#include <cstdlib>",
        "#include <cuda_runtime.h>",
        "#include <algorithm>",
        "#include <numeric>",
        "#include <vector>",
        "#define N 1024",
        "#define BLOCK 16",
        "#define ITERS 1000",
        "static void check(cudaError_t e, int line) {",
        '    if (e != cudaSuccess) { fprintf(stderr, "CUDA error line %d: %s\\n", line, cudaGetErrorString(e)); exit(1); }',
        "}",
        "#define CHECK(c) check((c), __LINE__)",
        "__global__ void matmul(const float*A, const float*B, float*C, int n) {",
        "    int r=blockIdx.y*blockDim.y+threadIdx.y, c=blockIdx.x*blockDim.x+threadIdx.x;",
        "    if (r<n && c<n) { float s=0; for(int k=0;k<n;++k) s+=A[r*n+k]*B[k*n+c]; C[r*n+c]=s; }",
        "}",
        "int main() {",
        "    size_t bytes = N*N*sizeof(float);",
        "    float *hA=(float*)malloc(bytes), *hB=(float*)malloc(bytes);",
        "    srand(42); for(int i=0;i<N*N;i++){hA[i]=(rand()%100)/100.f; hB[i]=(rand()%100)/100.f;}",
        "    float *dA, *dB, *dC;",
        "    CHECK(cudaMalloc(&dA,bytes)); CHECK(cudaMalloc(&dB,bytes)); CHECK(cudaMalloc(&dC,bytes));",
        "    CHECK(cudaMemcpy(dA,hA,bytes,cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(dB,hB,bytes,cudaMemcpyHostToDevice));",
        "    dim3 t(BLOCK,BLOCK), b((N+BLOCK-1)/BLOCK,(N+BLOCK-1)/BLOCK);",
        "    for(int i=0;i<10;i++){matmul<<<b,t>>>(dA,dB,dC,N);} CHECK(cudaDeviceSynchronize());",
        "    std::vector<float> times(ITERS);",
        "    for(int i=0;i<ITERS;i++){",
        "        cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));",
        "        CHECK(cudaEventRecord(s)); matmul<<<b,t>>>(dA,dB,dC,N); CHECK(cudaEventRecord(e));",
        "        CHECK(cudaEventSynchronize(e)); CHECK(cudaEventElapsedTime(&times[i],s,e));",
        "        CHECK(cudaEventDestroy(s)); CHECK(cudaEventDestroy(e));",
        "    }",
        "    std::sort(times.begin(),times.end());",
        "    float sum=std::accumulate(times.begin(),times.end(),0.0f);",
        "    float avg=sum/ITERS, p50=times[ITERS/2], p95=times[(int)(ITERS*0.95)], p99=times[(int)(ITERS*0.99)];",
        '    printf("%.4f %.4f %.4f %.4f %.4f %.4f\\n", avg, p50, p95, p99, times[0], times[ITERS-1]);',
        "    free(hA); free(hB); cudaFree(dA); cudaFree(dB); cudaFree(dC);",
        "}",
    ]
    return "\n".join(lines) + "\n"


@dataclass
class BenchResult:
    mhz: int
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    @property
    def gflops(self) -> float:
        return 2.0 * 1024**3 / (self.avg_ms * 1e6)

    @property
    def jitter_ms(self) -> float:
        return self.max_ms - self.min_ms


def _get_gpu_clock_range() -> tuple[int, int] | None:
    """Get min/max GPU clocks from nvidia-smi."""
    r = run("nvidia-smi -q -d CLOCK")
    if r.returncode != 0:
        return None
    min_clk = max_clk = 0
    for line in r.stdout.splitlines():
        if "Max Clocks" in line:
            # Next Graphics line has max
            continue
        m = re.match(r"\s+Graphics\s+:\s+(\d+)\s+MHz", line)
        if m:
            val = int(m.group(1))
            if max_clk == 0:
                min_clk = val  # first match is current (idle)
            max_clk = val  # keep updating, last "Max" section wins
    return (min_clk, max_clk) if max_clk > 0 else None


def _compile_bench(nvcc: str, tmpdir: str) -> str | None:
    """Compile the benchmark kernel, return binary path or None."""
    src = Path(tmpdir) / "autotune.cu"
    out = Path(tmpdir) / "autotune"
    src.write_text(_bench_kernel_source())
    r = run(f"{nvcc} -arch=sm_121 -O3 -o {out} {src}")
    if r.returncode != 0:
        print(red(f"  nvcc error: {r.stderr.strip()[-300:]}"))
        return None
    # Verify binary exists and is executable
    if not Path(out).exists():
        print(red(f"  Binary not created at {out}"))
        return None
    return str(out)


def _run_bench(binary: str, mhz: int) -> BenchResult | None:
    """Lock clocks, run benchmark, parse result. mhz=0 means DVFS auto."""
    if mhz > 0:
        run(f"nvidia-smi -lgc {mhz},{mhz}")
    time.sleep(0.5)  # settle after clock change
    r = run(binary)
    if r.returncode != 0:
        return None
    parts = r.stdout.strip().split()
    if len(parts) != 6:
        return None
    avg, p50, p95, p99, mn, mx = (float(x) for x in parts)
    return BenchResult(
        mhz=mhz, avg_ms=avg, p50_ms=p50, p95_ms=p95, p99_ms=p99, min_ms=mn, max_ms=mx
    )


def _get_current_locked_clock() -> int | None:
    """Read persisted clock from config file."""
    conf = Path(AUTOTUNE_CONF)
    if not conf.exists():
        return None
    text = conf.read_text().strip()
    m = re.match(r"(\d+)", text)
    return int(m.group(1)) if m else None


def _persist_clock(mhz: int) -> None:
    """Write config + systemd service to apply clock lock on boot."""
    Path(AUTOTUNE_CONF).write_text(f"{mhz}\n")
    _write_inference_tune_service(gpu_clock_mhz=mhz)


def _write_inference_tune_service(gpu_clock_mhz: int | None = None) -> None:
    """Write systemd service for all boot-time inference tuning."""
    # Migrate from old single-purpose service if present
    old = Path(AUTOTUNE_SERVICE)
    if old.exists() and str(old) != str(INFERENCE_TUNE_SERVICE):
        run("systemctl disable nvidia-gpu-clock.service 2>/dev/null")
        run("systemctl stop nvidia-gpu-clock.service 2>/dev/null")
        old.unlink(missing_ok=True)

    exec_lines: list[str] = []

    # GPU clock lock
    if gpu_clock_mhz is None:
        # Read from config if not specified
        conf = Path(AUTOTUNE_CONF)
        if conf.exists():
            m = re.match(r"(\d+)", conf.read_text().strip())
            if m:
                gpu_clock_mhz = int(m.group(1))
    if gpu_clock_mhz:
        exec_lines.append(f"ExecStart=/usr/bin/nvidia-smi -lgc {gpu_clock_mhz},{gpu_clock_mhz}")

    # CPU idle state disabling (LPI-1+ with latency > 0)
    exec_lines.append(
        "ExecStart=/bin/bash -c "
        "'for s in /sys/devices/system/cpu/cpu*/cpuidle/state*/; do "
        '[ "$(cat $s/latency 2>/dev/null)" != "0" ] && echo 1 > $s/disable 2>/dev/null; '
        "done; true'"
    )

    stop_lines: list[str] = []
    if gpu_clock_mhz:
        stop_lines.append("ExecStop=/usr/bin/nvidia-smi -rgc")

    Path(INFERENCE_TUNE_SERVICE).write_text(
        "[Unit]\n"
        "Description=Inference tuning (GPU clock lock + CPU idle states)\n"
        "After=nvidia-persistenced.service\n"
        "\n"
        "[Service]\n"
        "Type=oneshot\n"
        "RemainAfterExit=yes\n"
        + "\n".join(exec_lines)
        + "\n"
        + "\n".join(stop_lines)
        + ("\n" if stop_lines else "")
        + "\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )
    run("systemctl daemon-reload")
    run("systemctl enable inference-tune.service")
    run("systemctl start inference-tune.service")


def autotune_gpu_clock(apply: bool = False) -> None:
    """Sweep GPU clock speeds and find optimal frequency."""
    print(bold("\n  GPU Clock Autotune (1000-iter matmul benchmark)\n"))

    # Find nvcc
    nvcc_path = "/usr/local/cuda/bin/nvcc"
    r = run(f"{nvcc_path} --version")
    if r.returncode != 0:
        print(red("  nvcc not found — skipping autotune."))
        return

    # Check clock lock support
    clk_range = _get_gpu_clock_range()
    if not clk_range:
        print(red("  Cannot read GPU clock range — skipping autotune."))
        return

    _, max_clk = clk_range

    # Single tempdir for compilation — reused for sweep + DVFS baseline
    with tempfile.TemporaryDirectory(prefix="gpu-autotune-") as tmpdir:
        print("  Compiling benchmark kernel...")
        binary = _compile_bench(nvcc_path, tmpdir)
        if not binary:
            print(red("  Compilation failed — skipping autotune."))
            return

        # Determine sweep range: 100 MHz steps from max-500 to max
        step = 100
        # Below ~1500 MHz is power-saving territory, not useful for inference
        sweep_start = max(1500, max_clk - 500)
        sweep_start = (sweep_start // step) * step
        freqs = list(range(sweep_start, max_clk + 1, step))
        if max_clk not in freqs:
            freqs.append(max_clk)

        print(f"  Sweeping {len(freqs)} frequencies: {freqs[0]}-{freqs[-1]} MHz ({step} MHz steps)")
        print(f"  {len(freqs) * 1000} total iterations (~{len(freqs) * 2} min)\n")

        results: list[BenchResult] = []
        hdr = f"    {'MHz':>5s}  {'avg ms':>8s}  {'GFLOPS':>7s}  {'p50':>8s}  {'p95':>8s}  {'p99':>8s}  {'jitter':>7s}"
        print(dim(hdr))

        for mhz in freqs:
            br = _run_bench(binary, mhz)
            if br is None:
                print(f"    {mhz:5d}  FAILED")
                continue
            results.append(br)
            line = (
                f"    {br.mhz:5d}  {br.avg_ms:8.3f}  {br.gflops:7.0f}  "
                f"{br.p50_ms:8.3f}  {br.p95_ms:8.3f}  {br.p99_ms:8.3f}  {br.jitter_ms:7.3f}"
            )
            print(line)

        # DVFS auto — no warmup, to capture cold-start ramp-up cost
        run("nvidia-smi -rgc")
        time.sleep(2)  # let GPU settle back to idle
        print()
        print("  Running DVFS auto baseline (cold start)...")
        auto_br = _run_bench(binary, 0)  # 0 = DVFS auto, no clock lock
        run("nvidia-smi -rgc")  # ensure reset
        if auto_br:
            auto_br.mhz = 0
            line = (
                f"    {'auto':>5s}  {auto_br.avg_ms:8.3f}  {auto_br.gflops:7.0f}  "
                f"{auto_br.p50_ms:8.3f}  {auto_br.p95_ms:8.3f}  {auto_br.p99_ms:8.3f}  {auto_br.jitter_ms:7.3f}"
            )
            print(line)
            results.append(auto_br)

    if not results:
        print(red("\n  No valid results — autotune failed."))
        run("nvidia-smi -rgc")
        return

    # Pick winner: best avg GFLOPS, break ties by lowest p99
    best = min(results, key=lambda r: (r.avg_ms, r.p99_ms))
    # Also find best p99 for latency-sensitive recommendation
    best_p99 = min(results, key=lambda r: (r.p99_ms, r.avg_ms))

    print()
    if best.mhz == 0:
        print(green(f"  Best throughput: DVFS auto — {best.gflops:.0f} GFLOPS avg"))
    else:
        print(green(f"  Best throughput: {best.mhz} MHz — {best.gflops:.0f} GFLOPS avg"))
    if best_p99.mhz != best.mhz:
        label = "DVFS auto" if best_p99.mhz == 0 else f"{best_p99.mhz} MHz"
        print(f"  Best tail latency: {label} — p99 {best_p99.p99_ms:.3f} ms")

    # Reset clocks before deciding
    run("nvidia-smi -rgc")

    # If DVFS auto wins, recommend no lock
    if best.mhz == 0:
        current = _get_current_locked_clock()
        if current:
            print(yellow(f"\n  DVFS auto is optimal. Currently locked at {current} MHz."))
            if apply or ask("Remove clock lock and use DVFS auto?"):
                Path(AUTOTUNE_CONF).unlink(missing_ok=True)
                run("nvidia-smi -rgc")
                # Rewrite service without GPU clock, keeping CPU idle tuning
                _write_inference_tune_service(gpu_clock_mhz=None)
                print(green("  Clock lock removed. Using DVFS auto. CPU idle tuning preserved."))
        else:
            print(green("\n  DVFS auto is already optimal. No clock lock needed."))
        return

    # Recommend locking
    current = _get_current_locked_clock()
    if current == best.mhz:
        print(green(f"\n  Already locked at optimal {best.mhz} MHz."))
        return

    print()
    if apply or ask(f"Lock GPU clock to {best.mhz} MHz and persist across reboots?"):
        _persist_clock(best.mhz)
        print(green(f"  GPU clock locked to {best.mhz} MHz."))
        print(dim(f"    Config: {AUTOTUNE_CONF}"))
        print(dim(f"    Service: {AUTOTUNE_SERVICE} (enabled)"))
    else:
        print(dim("  Skipped."))


# ---------------------------------------------------------------------------
# Revert to stock
# ---------------------------------------------------------------------------
MANAGED_FILES = [
    SYSCTL_CONF,
    NM_POWERSAVE_CONF,
    MT7925_MODPROBE_CONF,
    MT7925_UDEV_RULE,
    LOGIND_NOSLEEP_CONF,
    AUTOTUNE_CONF,
    AUTOTUNE_SERVICE,
    INFERENCE_TUNE_SERVICE,
]


def revert_to_stock() -> None:
    """Remove all tuning applied by this script and restore defaults."""
    print(bold("\n=== Reverting to stock settings ===\n"))

    # 1. Sysctls
    conf = Path(SYSCTL_CONF)
    if conf.exists():
        conf.unlink()
        run("sysctl --system")
        print(green("  Removed sysctl overrides and reloaded defaults."))

    # 2. Docker — remove our keys from daemon.json, keep the rest
    daemon = Path(DOCKER_DAEMON_JSON)
    if daemon.exists():
        with contextlib.suppress(json.JSONDecodeError):
            cfg = json.loads(daemon.read_text())
            for key in EXPECTED_DOCKER_CONFIG:
                cfg.pop(key, None)
            if cfg:
                daemon.write_text(json.dumps(cfg, indent=4) + "\n")
            else:
                daemon.unlink()
            run("systemctl restart docker")
            print(green("  Removed inference tuning from Docker daemon.json."))

    # 3. WiFi
    for f in [NM_POWERSAVE_CONF, MT7925_MODPROBE_CONF, MT7925_UDEV_RULE]:
        p = Path(f)
        if p.exists():
            p.unlink()
    # Unpin WiFi band
    r = run("nmcli -t -f NAME connection show --active")
    conn_name = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else ""
    if conn_name:
        run(f"nmcli connection modify {shlex.quote(conn_name)} wifi.band ''")
    iface = _find_wifi_interface()
    if iface:
        run(f"iw dev {iface} set power_save on")
    run("systemctl restart NetworkManager")
    print(green("  WiFi power save re-enabled, MT7925 workarounds removed, band unpinned."))

    # 4. NVIDIA persistence — leave enabled, it's harmless and expected

    # 5. CPU governor — back to schedutil
    cpus = Path("/sys/devices/system/cpu/")
    for gov in sorted(cpus.glob("cpu*/cpufreq/scaling_governor")):
        gov.write_text("schedutil\n")
    print(green("  CPU governor set to schedutil."))

    # 5b. CPU idle — re-enable all states
    count = 0
    for disable in sorted(cpus.glob("cpu*/cpuidle/state*/disable")):
        if disable.read_text().strip() == "1":
            disable.write_text("0\n")
            count += 1
    if count:
        print(green(f"  Re-enabled {count} CPU idle states."))

    # 6. Sleep prevention — unmask targets, remove logind conf, reset GNOME
    for target in SLEEP_TARGETS:
        run(f"systemctl unmask {target}")
    Path(LOGIND_NOSLEEP_CONF).unlink(missing_ok=True)
    run("systemctl restart systemd-logind")
    r = run("which gsettings 2>/dev/null")
    if r.returncode == 0:
        for user in _desktop_users():
            schema = "org.gnome.settings-daemon.plugins.power"
            run(_gsettings_cmd(user, f"reset {schema} sleep-inactive-ac-type"))
            run(_gsettings_cmd(user, f"reset {schema} sleep-inactive-battery-type"))
            run(_gsettings_cmd(user, f"reset {schema} lid-close-ac-action"))
            run(_gsettings_cmd(user, f"reset {schema} lid-close-battery-action"))
            run(_gsettings_cmd(user, "reset org.gnome.desktop.session idle-delay"))
    print(green("  Sleep/suspend unmasked, logind and GNOME reset to defaults."))

    # 7. GPU clock lock — stop service, reset clocks
    for svc in ["inference-tune.service", "nvidia-gpu-clock.service"]:
        run(f"systemctl stop {svc} 2>/dev/null")
        run(f"systemctl disable {svc} 2>/dev/null")
    for f in [AUTOTUNE_CONF, AUTOTUNE_SERVICE, INFERENCE_TUNE_SERVICE]:
        Path(f).unlink(missing_ok=True)
    run("systemctl daemon-reload")
    run("nvidia-smi -rgc 2>/dev/null")
    print(green("  GPU clock lock removed, clocks reset to DVFS auto."))

    print(bold("\n  All tuning reverted to stock."))
    print(yellow("  Reboot recommended to fully restore defaults."))
    print(dim("    sudo reboot"))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="DGX Spark inference tuning validator")
    parser.add_argument("--check", action="store_true", help="Check only, no changes")
    parser.add_argument("--apply", action="store_true", help="Apply all fixes without prompting")
    parser.add_argument(
        "--autotune", action="store_true", help="Benchmark GPU clocks and lock optimal"
    )
    parser.add_argument(
        "--revert", action="store_true", help="Remove all tuning and restore stock settings"
    )
    args = parser.parse_args()

    if os.geteuid() != 0:
        print(red("Error: must run as root (sudo)."))
        sys.exit(1)

    if args.revert:
        revert_to_stock()
        return

    interactive = sys.stdin.isatty() and not args.check

    print(bold("\n=== DGX Spark Inference Tuning Validator ===\n"))

    sections: list[tuple[str, list[Result], Any]] = []

    # 1. Sysctls
    sysctl_results = check_sysctls()
    sysctl_results.append(check_tfo_key())
    sections.append(("Kernel sysctls (network + VM)", sysctl_results, write_sysctl_conf))

    # 2. Docker
    sections.append(("Docker + NVIDIA runtime", check_docker(), fix_docker))

    # 3. WiFi
    wifi_iface = _find_wifi_interface()
    if wifi_iface:
        sections.append(("WiFi tuning", check_wifi(wifi_iface), fix_wifi))
    else:
        sections.append(
            (
                "WiFi tuning",
                [Result("wifi interface", True, "none (ethernet only)", "N/A")],
                fix_wifi,
            )
        )

    # 4. NVIDIA
    sections.append(("NVIDIA GPU", check_nvidia(), fix_nvidia_persistence))

    # 5. CPU governor
    sections.append(("CPU governor", check_cpu_governor(), fix_cpu_governor))

    # 5b. CPU idle states
    idle_results = check_cpu_idle()
    if idle_results:
        sections.append(("CPU idle states", idle_results, fix_cpu_idle))

    # 6. CUDA toolkit
    sections.append(("CUDA toolkit", check_cuda(), fix_cuda))

    # 7. Sleep prevention
    sections.append(("Sleep / suspend prevention", check_sleep(), fix_sleep))

    # 8. GPU clock lock (informational — autotune sets this)
    locked = _get_current_locked_clock()
    # Check both old and new service names
    svc = run("systemctl is-active inference-tune.service 2>/dev/null")
    if svc.stdout.strip() != "active":
        svc = run("systemctl is-active nvidia-gpu-clock.service 2>/dev/null")
    svc_active = svc.stdout.strip() == "active"
    if locked:
        clock_results = [
            Result(
                name=f"GPU clock locked at {locked} MHz",
                ok=svc_active,
                current=f"{locked} MHz, service {'active' if svc_active else 'inactive'}",
                expected=f"{locked} MHz, service active",
            )
        ]
        sections.append(
            ("GPU clock lock (autotuned)", clock_results, lambda _mhz=locked: _persist_clock(_mhz))
        )

    # Display all results
    total_ok = 0
    total_fail = 0
    fixable_sections: list[tuple[str, Any]] = []

    for section_name, results, fix_fn in sections:
        print(bold(f"  {section_name}"))
        section_failed = False
        for r in results:
            r.display()
            if r.ok:
                total_ok += 1
            else:
                total_fail += 1
                section_failed = True
        if section_failed:
            fixable_sections.append((section_name, fix_fn))
        print()

    # Summary
    summary = f"  {total_ok + total_fail} checks: {green(f'{total_ok} passed')}"
    if total_fail:
        summary += f", {red(f'{total_fail} failed')}"
    print(summary)

    if not total_fail:
        print(green("\n  All settings are correctly configured.\n"))
    elif args.check:
        print(yellow(f"\n  {total_fail} setting(s) need attention. Run without --check to fix.\n"))
        sys.exit(1)
    else:
        # Offer fixes
        print()
        for section_name, fix_fn in fixable_sections:
            if args.apply or ask(f"Fix {section_name}?"):
                print(f"  Applying: {section_name}...")
                try:
                    fix_fn()
                    print(green("  Done."))
                except Exception as e:
                    print(red(f"  Error: {e}"))
            else:
                print(dim(f"  Skipped: {section_name}"))
            print()

    # --- Migrate old nvidia-gpu-clock.service to inference-tune.service ---
    if not args.check:
        old_svc = Path(AUTOTUNE_SERVICE)
        new_svc = Path(INFERENCE_TUNE_SERVICE)
        if old_svc.exists() and not new_svc.exists():
            _write_inference_tune_service()
            print(dim("  Migrated nvidia-gpu-clock.service -> inference-tune.service"))

    # --- Interactive: hostname ---
    if not args.check and interactive:
        current_host = run("hostname").stdout.strip()
        current_fqdn = run("hostname -f").stdout.strip()
        print(bold("  Hostname"))
        print(f"    Current: {current_host} ({current_fqdn})")
        if ask("Change hostname?"):
            hostname_re = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?$")
            new_short = input("    Short hostname: ").strip()
            new_fqdn = input("    FQDN (e.g. host.example.com): ").strip()
            if not new_short or not new_fqdn:
                print(dim("  Skipped (empty input)."))
            elif not hostname_re.match(new_short) or not hostname_re.match(new_fqdn):
                print(red("  Invalid hostname — only alphanumerics, hyphens, dots allowed."))
            else:
                run(f"hostnamectl set-hostname {shlex.quote(new_short)}")
                # Update /etc/hosts — replace old hostname line or append
                hosts = Path("/etc/hosts")
                lines = hosts.read_text().splitlines()
                new_lines = []
                replaced = False
                for line in lines:
                    if line.startswith("127.0.0.1") and current_host in line:
                        new_lines.append(f"127.0.0.1       {new_fqdn} {new_short}")
                        replaced = True
                    else:
                        new_lines.append(line)
                if not replaced:
                    new_lines.append(f"127.0.0.1       {new_fqdn} {new_short}")
                hosts.write_text("\n".join(new_lines) + "\n")
                print(green(f"  Hostname set to {new_short} ({new_fqdn})"))
        print()

    # --- Drop caches ---
    if not args.check:
        if args.apply or ask("Drop filesystem caches now to free memory for GPU?"):
            run("sync")
            Path("/proc/sys/vm/drop_caches").write_text("3\n")
            print(green("  Caches dropped."))
        print()

    # --- GPU clock autotune ---
    if args.autotune:
        autotune_gpu_clock(apply=args.apply)
    elif (
        not args.check
        and interactive
        and ask("Run GPU clock autotune? (~12 min, 1000 iters per frequency)")
    ):
        autotune_gpu_clock(apply=False)

    # --- System summary ---
    print(bold("  System Summary"))
    print()

    # Hostname
    hostname = run("hostname").stdout.strip()
    fqdn = run("hostname -f").stdout.strip()
    print(f"    Host:     {bold(hostname)} ({fqdn})")

    # CPU
    cpu_r = run("lscpu")
    cpu_model = ""
    cpu_cores = ""
    for line in cpu_r.stdout.splitlines():
        if line.startswith("Model name:"):
            cpu_model = line.split(":", 1)[1].strip()
        elif line.startswith("CPU(s):"):
            cpu_cores = line.split(":", 1)[1].strip()
    if cpu_model:
        print(f"    CPU:      {cpu_model} ({cpu_cores} cores)")

    # Memory
    meminfo: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            meminfo[parts[0].rstrip(":")] = int(parts[1])
    mem_total = meminfo.get("MemTotal", 0)
    mem_free = meminfo.get("MemFree", 0)
    mem_avail = meminfo.get("MemAvailable", 0)
    cached = meminfo.get("Cached", 0)
    buffers = meminfo.get("Buffers", 0)
    slab_reclaim = meminfo.get("SReclaimable", 0)
    cache_total = cached + buffers + slab_reclaim

    def _gb(kb: int) -> str:
        return f"{kb / 1048576:.1f} GB"

    print(f"    Memory:   {_gb(mem_total)} total, {_gb(mem_free)} free, {_gb(mem_avail)} available")
    print(
        f"    Cache:    {_gb(cache_total)} (buffers {_gb(buffers)}, cached {_gb(cached)}, slab {_gb(slab_reclaim)})"
    )

    # GPU
    gpu_r = run(
        "nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,power.draw --format=csv,noheader,nounits"
    )
    if gpu_r.returncode == 0 and gpu_r.stdout.strip():
        parts = [p.strip() for p in gpu_r.stdout.strip().split(",")]
        if len(parts) >= 6:
            gpu_name, gpu_mem_total, gpu_mem_used, gpu_mem_free, gpu_temp, gpu_power = parts[:6]
            print(f"    GPU:      {gpu_name}")
            if gpu_mem_total.replace(" ", "").isdigit():
                print(
                    f"    GPU mem:  {gpu_mem_total} MB total, {gpu_mem_used} MB used, {gpu_mem_free} MB free"
                )
            else:
                print(f"    GPU mem:  unified with system RAM ({_gb(mem_total)} shared)")
            print(f"    GPU temp: {gpu_temp}C, power: {gpu_power}W")

    # CUDA
    nvcc = _nvcc_version()
    drv_r = run("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits")
    driver = drv_r.stdout.strip() if drv_r.returncode == 0 else "unknown"
    print(f"    CUDA:     {nvcc or 'not found'} (driver {driver})")

    # Uptime
    up_r = run("uptime -p")
    print(f"    Uptime:   {up_r.stdout.strip()}")

    print()
    print(yellow("  Recommendation: reboot to ensure all settings take full effect."))
    print(dim("    sudo reboot"))
    print()


if __name__ == "__main__":
    main()
