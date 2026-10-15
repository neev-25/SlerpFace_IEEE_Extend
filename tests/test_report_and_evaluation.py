"""Regression checks for research-report tooling and synthetic fixtures."""

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def test_colormap_registry_and_plot_cleanup(tmp_path, monkeypatch):
    report = importlib.import_module("generate_report_images")
    mapped = report._apply_cmap(np.arange(6).reshape(2, 3))
    assert mapped.shape == (2, 3, 3)
    assert mapped.dtype == np.uint8
    fig, _ = report.plt.subplots()
    number = fig.number

    def fail_save(*args, **kwargs):
        raise OSError("simulated output failure")

    monkeypatch.setattr(fig, "savefig", fail_save)
    with pytest.raises(OSError, match="simulated"):
        report._save_fig(fig, str(tmp_path / "fail.png"))
    assert not report.plt.fignum_exists(number)


def test_report_help_is_ascii_for_windows_console(capsys):
    report = importlib.import_module("generate_report_images")
    with pytest.raises(SystemExit) as exc:
        report.main(["--help"])
    assert exc.value.code == 0
    capsys.readouterr().out.encode("ascii")


def test_sample_import_has_no_disk_write(monkeypatch):
    from pathlib import Path
    from PIL import Image

    def forbidden(*args, **kwargs):
        raise AssertionError("import attempted disk write")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Image.Image, "save", forbidden)
    sys.modules.pop("create_demo_samples", None)
    importlib.import_module("create_demo_samples")


def test_sample_generation_is_reproducible_and_preserves_existing_files(tmp_path):
    samples = importlib.import_module("create_demo_samples")
    first = samples.generate_demo_samples(seed=17)
    second = samples.generate_demo_samples(seed=17)
    assert len(first) == 5
    for name in first:
        assert np.array_equal(first[name], second[name])
        assert first[name].shape == (112, 112, 3)
    paths = samples.write_demo_samples(tmp_path, seed=17)
    before = {path.name: path.read_bytes() for path in paths}
    with pytest.raises(FileExistsError):
        samples.write_demo_samples(tmp_path, seed=18)
    assert before == {path.name: path.read_bytes() for path in paths}


def test_synthetic_generators_do_not_change_global_rng():
    samples = importlib.import_module("create_demo_samples")
    evaluation = importlib.import_module("evaluate_antispoof")
    report = importlib.import_module("generate_report_images")
    before = np.random.get_state()
    samples.generate_demo_samples(seed=12)
    first = evaluation.generate_synthetic_benchmark_dataset(2, seed=12)
    second = evaluation.generate_synthetic_benchmark_dataset(2, seed=12)
    report.make_synthetic_face(seed=12)
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(before, after))
    assert len(first) == 6
    assert all(np.array_equal(a["img"], b["img"]) for a, b in zip(first, second))


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_synthetic_dataset_rejects_invalid_count(count):
    evaluation = importlib.import_module("evaluate_antispoof")
    with pytest.raises(ValueError, match="positive integer"):
        evaluation.generate_synthetic_benchmark_dataset(count)


def test_report_rejects_before_extracting_or_writing_intermediates(tmp_path, monkeypatch):
    report = importlib.import_module("generate_report_images")
    calls = []
    monkeypatch.setattr(report, "stage1_input", lambda *args: calls.append("input"))
    monkeypatch.setattr(report, "stage2_pad", lambda *args, **kwargs: SimpleNamespace(is_live=False))

    def forbidden(*args, **kwargs):
        raise AssertionError("Rejected input reached extraction")

    monkeypatch.setattr(report, "extract_face_features", forbidden)
    assert report.main(["--demo", "--out", str(tmp_path)]) == 2
    assert calls == ["input"]
    assert list(tmp_path.iterdir()) == []


def test_report_uses_source_pixels_and_one_context(tmp_path, monkeypatch):
    report = importlib.import_module("generate_report_images")
    from PIL import Image

    source = np.full((37, 59, 3), [157, 101, 87], dtype=np.uint8)
    path = tmp_path / "input.png"
    Image.fromarray(source).save(path)
    calls = []
    contexts = []
    monkeypatch.setattr(report, "stage1_input", lambda *args: None)

    def stage2(image, directory, *, demo):
        assert demo is True
        assert np.array_equal(image, source)
        return SimpleNamespace(is_live=True)

    def extract(image, model_path, *, demo):
        assert demo is True
        assert np.array_equal(image, source)
        calls.append("extracted")
        return np.ones((49, 16)), np.ones(512)

    monkeypatch.setattr(report, "stage2_pad", stage2)
    monkeypatch.setattr(report, "extract_face_features", extract)
    monkeypatch.setattr(report, "stage3_feature_extract", lambda *args, **kwargs: None)
    monkeypatch.setattr(report, "stage4_slerp", lambda *args, **kwargs: contexts.append(kwargs["context"]))
    monkeypatch.setattr(report, "stage5_drop", lambda *args, **kwargs: contexts.append(kwargs["context"]))
    monkeypatch.setattr(report, "stage6_output", lambda *args, **kwargs: None)
    monkeypatch.setattr(report, "summary_strip", lambda *args: None)
    output = tmp_path / "report"
    assert report.main(["--img", str(path), "--demo", "--out", str(output)]) == 0
    assert calls == ["extracted"]
    assert contexts[0] is contexts[1]
    assert list(output.iterdir()) == []


def test_real_report_uses_aligned_vector_groups_and_does_not_gate_on_texture(tmp_path, monkeypatch):
    report = importlib.import_module("generate_report_images")
    from PIL import Image

    path = tmp_path / "capture.png"
    Image.new("RGB", (640, 480)).save(path)
    face = SimpleNamespace(aligned_rgb=np.full((112, 112, 3), 80, dtype=np.uint8),
                           groups=np.arange(1, 129, dtype=float).reshape(8, 16))
    monkeypatch.setattr(report, "extract_pretrained_face", lambda image: face)
    monkeypatch.setattr(report, "stage1_input", lambda *args: None)
    monkeypatch.setattr(report, "stage2_pad", lambda *args, **kwargs: SimpleNamespace(is_live=False))
    received = []
    monkeypatch.setattr(report, "stage3_feature_extract", lambda image, groups, *args, **kwargs: received.append((image, groups)))
    for stage in ("stage4_slerp", "stage5_drop", "stage6_output", "summary_strip"):
        monkeypatch.setattr(report, stage, lambda *args, **kwargs: None)
    assert report.main(["--img", str(path), "--out", str(tmp_path / "real-report")]) == 0
    assert received[0][1].shape == (8, 16)
    np.testing.assert_array_equal(received[0][0], face.aligned_rgb)


def test_sface_report_figure_accepts_global_embedding_groups(tmp_path):
    report = importlib.import_module("generate_report_images")
    report.stage3_feature_extract(np.zeros((112, 112, 3), dtype=np.uint8),
                                  np.arange(128, dtype=float).reshape(8, 16), str(tmp_path))
    assert (tmp_path / "stage3_feature_extraction.png").stat().st_size > 0
    assert report.plt.get_fignums() == []


@pytest.mark.parametrize("option,value", [("--alpha", "1"), ("--alpha", "nan"),
                                          ("--drop", "0.95"), ("--drop", "-0.1"),
                                          ("--camera-id", "-1"), ("--seed", "-1")])
def test_report_invalid_parameters_fail_before_output(tmp_path, option, value):
    report = importlib.import_module("generate_report_images")
    output = tmp_path / "report"
    with pytest.raises(SystemExit) as exc:
        report.main(["--demo", "--out", str(output), option, value])
    assert exc.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("kind", ["corrupt", "tiny"])
def test_report_bad_input_fails_before_output(tmp_path, kind):
    report = importlib.import_module("generate_report_images")
    from PIL import Image

    path = tmp_path / "source.png"
    if kind == "corrupt":
        path.write_bytes(b"not an image")
    else:
        Image.new("RGB", (2, 2)).save(path)
    output = tmp_path / "report"
    with pytest.raises(SystemExit) as exc:
        report.main(["--img", str(path), "--demo", "--out", str(output)])
    assert exc.value.code == 2
    assert not output.exists()


def test_report_failure_closes_figures_and_leaves_no_temp_image(tmp_path, monkeypatch):
    report = importlib.import_module("generate_report_images")

    def fail_stage(*args, **kwargs):
        report.plt.figure()
        raise RuntimeError("injected plot failure")

    monkeypatch.setattr(report, "stage1_input", fail_stage)
    assert report.main(["--demo", "--out", str(tmp_path)]) == 1
    assert report.plt.get_fignums() == []
    assert list(tmp_path.iterdir()) == []


def test_evaluation_prints_only_measured_synthetic_results(capsys):
    evaluation = importlib.import_module("evaluate_antispoof")
    result = evaluation.evaluate_system(num_samples_per_class=2, seed=5)
    text = capsys.readouterr().out
    assert result["synthetic_only"] is True
    assert result["authentication_validated"] is False
    assert len(result["classes"]) == 3
    assert all(item["count"] == 2 for item in result["classes"].values())
    assert "SYNTHETIC DEMO" in text
    assert "Zero-Leakage" not in text
    assert "SlerpFace Alone" not in text
    assert "Privacy Guarantee" not in text


def test_webcam_capture_excludes_preview_overlay_and_releases_device(monkeypatch):
    report = importlib.import_module("generate_report_images")
    raw = np.arange(8 * 6 * 3, dtype=np.uint8).reshape(8, 6, 3)
    state = {"released": False, "closed": False}

    class Camera:
        def isOpened(self):
            return True

        def set(self, *args):
            pass

        def read(self):
            return True, raw.copy()

        def release(self):
            state["released"] = True

    def annotate(preview, *args):
        preview[:] = 255

    fake_cv2 = SimpleNamespace(
        VideoCapture=lambda index: Camera(), CAP_PROP_FRAME_WIDTH=1, CAP_PROP_FRAME_HEIGHT=2,
        FONT_HERSHEY_SIMPLEX=0, COLOR_BGR2RGB=3,
        flip=lambda frame, direction: np.fliplr(frame).copy(), rectangle=annotate,
        addWeighted=lambda overlay, a, preview, b, c, destination: None,
        putText=annotate, imshow=lambda *args: None, waitKey=lambda delay: 13,
        cvtColor=lambda frame, code: frame[:, :, ::-1].copy(),
        destroyAllWindows=lambda: state.update(closed=True),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    actual = report.capture_from_webcam()
    assert np.array_equal(actual, raw[:, :, ::-1])
    assert state == {"released": True, "closed": True}


def test_webcam_capture_releases_device_on_failure(monkeypatch):
    report = importlib.import_module("generate_report_images")
    state = {"released": False, "closed": False}

    class Camera:
        def isOpened(self):
            return True

        def set(self, *args):
            raise RuntimeError("camera setup failed")

        def release(self):
            state["released"] = True

    fake_cv2 = SimpleNamespace(
        VideoCapture=lambda index: Camera(), CAP_PROP_FRAME_WIDTH=1, CAP_PROP_FRAME_HEIGHT=2,
        destroyAllWindows=lambda: state.update(closed=True),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    with pytest.raises(RuntimeError, match="camera setup"):
        report.capture_from_webcam()
    assert state == {"released": True, "closed": True}
