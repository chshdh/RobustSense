from robustsense.experiments.v2_phase0 import digest_entries, sha256_file, verify_file_entries


def test_v2_manifest_digest_is_order_independent(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")
    entries = [
        {"path": "second.txt", "bytes": 4, "sha256": sha256_file(second)},
        {"path": "first.txt", "bytes": 5, "sha256": sha256_file(first)},
    ]
    assert digest_entries(entries) == digest_entries(reversed(entries))
    assert verify_file_entries(tmp_path, entries) == []


def test_v2_manifest_verification_detects_changed_file(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("before", encoding="utf-8")
    entries = [{"path": "artifact.txt", "bytes": 6, "sha256": sha256_file(artifact)}]
    artifact.write_text("after!", encoding="utf-8")
    problems = verify_file_entries(tmp_path, entries)
    assert len(problems) == 1
    assert problems[0].startswith("sha256:artifact.txt:")

