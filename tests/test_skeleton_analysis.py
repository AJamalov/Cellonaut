from __future__ import annotations

from cellonaut.masks import skeleton_analysis


class FakeMask:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeResult:
    def getBranches(self):
        return [4, 7]

    def getEndPoints(self):
        return [2, 3]

    def getJunctions(self):
        return [1, 2]


class FakeAnalyzer:
    NONE = 0
    run_args = None

    def setup(self, _arg, image):
        self.image = image

    def run(self, *args):
        type(self).run_args = args
        return FakeResult()


class FakeDuplicator:
    def __init__(self, duplicate):
        self.duplicate = duplicate

    def run(self, _image):
        return self.duplicate


class FakeIJ:
    def __init__(self):
        self.calls = []

    def run(self, image, command, options):
        self.calls.append((image, command, options))


def test_analyze_mask_skeleton_sums_tree_metrics_and_closes_duplicate(monkeypatch):
    duplicate = FakeMask()
    ij = FakeIJ()
    java_booleans = []

    def fake_java_boolean(value):
        typed_value = ("JBoolean", value)
        java_booleans.append(typed_value)
        return typed_value

    monkeypatch.setattr(
        skeleton_analysis,
        "get_java_classes",
        lambda: {
            "Duplicator": lambda: FakeDuplicator(duplicate),
            "IJ": ij,
        },
    )
    monkeypatch.setattr(skeleton_analysis, "jimport", lambda _name: FakeAnalyzer)
    monkeypatch.setattr(skeleton_analysis, "JBoolean", fake_java_boolean)

    metrics = skeleton_analysis.analyze_mask_skeleton(FakeMask())

    assert metrics == {
        "SkeletonCount": 2,
        "BranchCount": 11,
        "EndpointCount": 5,
        "JunctionCount": 3,
    }
    assert FakeAnalyzer.run_args == (
        0,
        ("JBoolean", False),
        ("JBoolean", False),
        None,
        ("JBoolean", True),
        ("JBoolean", False),
    )
    assert java_booleans == [
        ("JBoolean", False),
        ("JBoolean", False),
        ("JBoolean", True),
        ("JBoolean", False),
    ]
    assert ij.calls == [(duplicate, "Skeletonize", "")]
    assert duplicate.closed is True


def test_analyze_mask_skeleton_saves_generated_skeleton(monkeypatch, tmp_path):
    duplicate = FakeMask()
    saved = []
    file_saver = object()
    monkeypatch.setattr(
        skeleton_analysis,
        "get_java_classes",
        lambda: {
            "Duplicator": lambda: FakeDuplicator(duplicate),
            "IJ": FakeIJ(),
            "FileSaver": file_saver,
        },
    )
    monkeypatch.setattr(skeleton_analysis, "jimport", lambda _name: FakeAnalyzer)
    monkeypatch.setattr(skeleton_analysis, "JBoolean", lambda value: value)
    monkeypatch.setattr(
        skeleton_analysis,
        "save_imagej_tiff",
        lambda path, image, file_saver: saved.append(
            (path, image, file_saver)
        ) or True,
    )
    output_path = tmp_path / "SampleA_Tubules_skeleton.tif"

    skeleton_analysis.analyze_mask_skeleton(
        FakeMask(),
        skeleton_path=output_path,
    )

    assert saved == [(output_path, duplicate, file_saver)]
    assert duplicate.closed is True
