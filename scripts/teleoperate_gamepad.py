"""官方 LeRobot 游戏手柄 -> 本机 MuJoCo；手柄与 Viewer 各占一个 GUI 进程。"""

from __future__ import annotations

import argparse
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from so_ws.teleoperators.gamepad_ik import DEFAULT_IK, IK_MODES


def run_worker(ik_mode):
    import mujoco.viewer

    from so_ws.teleoperators.gamepad_mujoco import GamepadMujoco

    if ik_mode == "official":
        from so_ws.teleoperators.official_gamepad_mujoco import OfficialGamepadMujoco

        sim = OfficialGamepadMujoco()
    else:
        sim = GamepadMujoco(ik_mode=ik_mode)
    pending = b""
    last_reason = None
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.lookat[:] = [0.18, 0.02, 0.10]
        viewer.cam.distance = 0.8
        print(json.dumps({"ready": True, "dt": sim.dt, "ik": sim.ik_mode}), flush=True)
        while viewer.is_running():
            started = time.monotonic()
            # os.read 避免 TextIOWrapper 缓冲区与 select 不一致；每拍只应用最后一个新鲜包。
            latest = None
            while select.select([sys.stdin], [], [], 0)[0]:
                chunk = os.read(sys.stdin.fileno(), 65536)
                if not chunk:
                    return
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    latest = json.loads(line)
            if latest is not None:
                try:
                    sim.receive(latest, started)
                except ValueError as error:
                    print(f"输入被拒绝：{error}", file=sys.stderr)
            sim.advance(started)
            if sim.reason != last_reason:
                print(sim.reason, file=sys.stderr)
                last_reason = sim.reason
            viewer.sync()
            time.sleep(max(0.0, sim.dt - (time.monotonic() - started)))


class CheckedHIDReader:
    """为 F310 D 模式补齐官方解析器预期的 report ID，并显式传播拔线错误。"""

    def __init__(self, device):
        self.device = device

    def read(self, size):
        try:
            report = self.device.read(size)
        except OSError as error:
            raise RuntimeError("手柄 HID 连接断开") from error
        if report and len(report) != 8:
            raise RuntimeError(f"F310 HID 报文长度异常：{len(report)}")
        # 实测 F310: [LX, LY, RX, RY, hat/ABXY, shoulders/misc, ..., ...]。
        # 官方 RumblePad 解析器从 data[1] 起读取轴，需在本地补一个 report ID。
        return [0, *report] if report else []

    def close(self):
        self.device.close()


def connect_gamepad():
    import pygame
    from lerobot.teleoperators.gamepad import GamepadTeleop, GamepadTeleopConfig

    # 本机 F310 D 模式在 SDL 2.28.4 中无法枚举，官方 HID 备用接口可以直接读取。
    teleop = GamepadTeleop(GamepadTeleopConfig(id="f310_mujoco", hidapi_fallback=sys.platform == "darwin"))
    try:
        teleop.connect()
        pad = teleop.gamepad
        if pad is None or not pad.running:
            raise RuntimeError("官方接口没有检测到手柄")
        if teleop.hidapi_fallback:
            info = pad.device_info
            if pad.device is None or (info["vendor_id"], info["product_id"]) != (0x046D, 0xC216):
                raise RuntimeError("本机 HID 入口要求 F310 的 D 模式 (046D:C216)")
            pad.device = CheckedHIDReader(pad.device)
            print(f"设备：{info['product_string']}；官方 HID 接口；046D:C216")
        else:
            if pad.joystick is None or pygame.joystick.get_count() != 1:
                raise RuntimeError("官方接口默认选择第一个设备，请只保留一个游戏手柄")
            if pad.joystick.get_numaxes() < 4 or pad.joystick.get_numbuttons() < 8:
                raise RuntimeError("当前设备不满足官方四轴、八按钮映射")
            print(f"设备：{pad.joystick.get_name()}；GUID：{pad.joystick.get_guid()}")
    except Exception:
        teleop.disconnect()
        raise
    return teleop


def wait_worker_ready(worker, timeout=30):
    """只接受带策略名的 ready 消息；允许第三方库在启动时输出诊断文本。"""
    deadline, pending = time.monotonic() + timeout, b""
    while time.monotonic() < deadline:
        if not select.select([worker.stdout], [], [], max(0.0, deadline - time.monotonic()))[0]:
            break
        chunk = os.read(worker.stdout.fileno(), 65536)
        if not chunk:
            raise RuntimeError("MuJoCo 子进程在就绪前退出，详见终端错误")
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            try:
                message = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                print(line.decode(errors="replace"), file=sys.stderr)
                continue
            if isinstance(message, dict) and message.get("ready"):
                return message
    raise RuntimeError("MuJoCo Viewer 启动超时")


def run_gamepad(ik_mode):
    # 输入与 Viewer 各占一个 GUI 进程；HID 输入不需要初始化 SDL 的手柄子系统。
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
    import pygame
    from lerobot.teleoperators.utils import TeleopEvents

    teleop = connect_gamepad()
    pad = teleop.gamepad
    worker = None
    try:
        print(f"IK 策略：{ik_mode} — {IK_MODES[ik_mode]}", flush=True)
        print("左摇杆：XY；右摇杆上下：Z；LT：关闭、RT：打开夹爪；按住 RB 移动。")
        if ik_mode != "official":
            print("先松开 RB、摇杆回中，再接管。")
        print("Esc 或关闭任意窗口退出；本入口仅控制仿真。")
        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode((720, 190))
        pygame.display.set_caption(f"SO-101 | IK: {ik_mode} | Simulation only")
        font = pygame.font.Font(None, 25)
        worker = subprocess.Popen(
            [str(ROOT / ".venv/bin/mjpython"), str(Path(__file__).resolve()), "--worker", "--ik", ik_mode],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        message = wait_worker_ready(worker)
        if message.get("ik") != ik_mode:
            raise RuntimeError(f"IK 策略不匹配：要求 {ik_mode}，子进程返回 {message.get('ik')}")
        dt = message["dt"]
        seq = 0
        while worker.poll() is None:
            started = time.monotonic()
            if pygame.event.peek(pygame.QUIT) or pygame.event.peek(pygame.JOYDEVICEREMOVED):
                break
            # 直接使用官方动作与接管事件，不维护第二套手柄轴/夹爪解析。
            action = teleop.get_action()
            events = teleop.get_teleop_events()
            if not teleop.hidapi_fallback:
                if pygame.joystick.get_count() != 1:
                    raise RuntimeError("手柄已断开，仿真控制结束")
                # 官方 get_deltas() 会吞掉拔线后的读取错误；这里让原设备断开显式退出。
                pad.joystick.get_axis(0)
            if pygame.key.get_pressed()[pygame.K_ESCAPE]:
                break
            enabled = bool(events[TeleopEvents.IS_INTERVENTION])
            packet = {
                "seq": seq,
                "stamp": time.monotonic(),
                "enabled": enabled,
                "action": {key: float(value) for key, value in action.items()},
            }
            worker.stdin.write(json.dumps(packet, allow_nan=False) + "\n")
            worker.stdin.flush()
            screen.fill((24, 28, 36))
            labels = [
                "Official LeRobot -> local MuJoCo | simulation only",
                f"IK: {ik_mode}",
                "Hold RB to move | release RB to pause | Esc to exit",
                f"Input request: {'RB held' if enabled else 'paused'}  XYZ: "
                + ", ".join(f"{action[f'delta_{axis}']:+.2f}" for axis in "xyz"),
            ]
            for index, label in enumerate(labels):
                screen.blit(font.render(label, True, (225, 232, 240)), (15, 20 + index * 38))
            pygame.display.flip()
            seq += 1
            time.sleep(max(0.0, dt - (time.monotonic() - started)))
        if worker.poll() not in (None, 0):
            raise RuntimeError(f"MuJoCo 子进程异常退出：{worker.returncode}")
    finally:
        if worker is not None:
            if worker.stdin is not None:
                try:
                    worker.stdin.close()
                except BrokenPipeError:
                    pass
            try:
                worker.wait(timeout=3)
            except subprocess.TimeoutExpired:
                worker.terminate()
                try:
                    worker.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
            if worker.stdout is not None:
                worker.stdout.close()
        teleop.disconnect()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ik",
        choices=IK_MODES,
        default=DEFAULT_IK,
        help="IK 策略（默认 official）；" + "；".join(f"{key}: {value}" for key, value in IK_MODES.items()),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.worker:
            run_worker(args.ik)
        else:
            run_gamepad(args.ik)
    except KeyboardInterrupt:
        pass
    except (RuntimeError, ValueError, OSError) as error:
        print(f"已停止：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
