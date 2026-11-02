"""SO-101 cube control panel, running outside mjpython's Cocoa process."""

from __future__ import annotations

import argparse
import json
import socket
import tkinter as tk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("socket_path")
    ap.add_argument("--x", type=float, required=True)
    ap.add_argument("--y", type=float, required=True)
    ap.add_argument("--z", type=float, required=True)
    args = ap.parse_args()

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(args.socket_path)
    sock.setblocking(False)
    send_buffer = []

    def send(message: dict) -> None:
        try:
            sock.sendall((json.dumps(message, ensure_ascii=False) + "\n").encode())
        except OSError:
            root.destroy()

    root = tk.Tk()
    root.title("SO-101 Cube 控制面板")
    root.geometry("460x330")
    root.resizable(False, False)
    root.lift()
    root.focus_force()

    state = {"x": args.x, "y": args.y, "z": args.z}
    variables = {}
    scales = {}
    updating = False

    tk.Label(root, text="Cube 位置控制", font=("Helvetica", 16, "bold")).pack(pady=(12, 2))
    tk.Label(root, text="摆放阶段拖动滑块；点击开始后锁定位置").pack(pady=(0, 8))

    ranges = {"x": (0.00, 0.40), "y": (-0.30, 0.30), "z": (0.015, 0.25)}
    labels = {"x": "X (m)", "y": "Y (m)", "z": "Z (m)"}
    for axis in ("x", "y", "z"):
        row = tk.Frame(root)
        row.pack(fill="x", padx=12, pady=2)
        tk.Label(row, text=labels[axis], width=7, anchor="w").pack(side="left")
        variable = tk.DoubleVar(value=state[axis])
        variables[axis] = variable

        def on_scale(value: str, axis: str = axis) -> None:
            if updating:
                return
            state[axis] = float(value)
            send({"type": "pose", **state})

        low, high = ranges[axis]
        scale = tk.Scale(
            row,
            from_=low,
            to=high,
            resolution=0.001,
            orient="horizontal",
            length=330,
            showvalue=True,
            variable=variable,
            command=on_scale,
        )
        scale.pack(side="left", fill="x", expand=True)
        scales[axis] = scale

    status = tk.StringVar(value="等待摆放")
    tk.Label(root, textvariable=status, anchor="w").pack(fill="x", padx=14, pady=(8, 4))
    buttons = tk.Frame(root)
    buttons.pack(pady=4)
    start_button = tk.Button(
        buttons,
        text="开始任务",
        width=16,
        command=lambda: send({"type": "start"}),
    )
    start_button.pack(side="left", padx=6)
    reset_button = tk.Button(
        buttons,
        text="重置并重新摆放",
        width=16,
        command=lambda: send({"type": "reset"}),
    )
    reset_button.pack(side="left", padx=6)

    def apply_parent_message(message: dict) -> None:
        nonlocal updating
        if message.get("type") == "phase":
            running = message.get("phase") == "running"
            for scale in scales.values():
                scale.configure(state=tk.DISABLED if running else tk.NORMAL)
            start_button.configure(state=tk.DISABLED if running else tk.NORMAL)
            status.set("策略执行中" if running else "等待摆放")
        elif message.get("type") == "pose":
            updating = True
            try:
                for axis in ("x", "y", "z"):
                    state[axis] = float(message[axis])
                    variables[axis].set(state[axis])
            finally:
                updating = False
        elif message.get("type") == "close":
            root.destroy()

    def poll_parent() -> None:
        try:
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    root.destroy()
                    return
                send_buffer.append(chunk.decode())
        except BlockingIOError:
            pass
        except OSError:
            root.destroy()
            return
        if send_buffer:
            text = "".join(send_buffer)
            send_buffer.clear()
            for line in text.splitlines():
                if line:
                    apply_parent_message(json.loads(line))
        root.after(50, poll_parent)

    def close_panel() -> None:
        try:
            send({"type": "close"})
            sock.close()
        finally:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_panel)
    root.after(50, poll_parent)
    root.mainloop()


if __name__ == "__main__":
    main()
