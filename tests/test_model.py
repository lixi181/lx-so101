from so_ws.robots.so101_mujoco.model import load_model


def test_so101_model_loads() -> None:
    model, data = load_model()
    assert model.nq == 6
    assert model.nu == 6
    assert data.time == 0.0
