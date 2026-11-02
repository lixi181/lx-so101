import mujoco

from so_ws.robots.so101_mujoco.model import load_model


def main() -> None:
    model, data = load_model()
    for _ in range(100):
        mujoco.mj_step(model, data)
    print(f"nq={model.nq} nu={model.nu} time={data.time:.3f}")


if __name__ == "__main__":
    main()
