import io
from pathlib import Path

from oas_core.storage import LocalStorage


def test_put_get_bytes(tmp_path: Path) -> None:
    fs = LocalStorage(tmp_path)
    uri = fs.uri_for("a/b/hello.txt")
    obj = fs.put_bytes(uri, b"hello")
    assert obj.size == 5
    assert obj.sha256 is not None
    assert fs.exists(uri)
    assert fs.get_bytes(uri) == b"hello"


def test_put_stream_and_list(tmp_path: Path) -> None:
    fs = LocalStorage(tmp_path)
    uri1 = fs.uri_for("dir/one.bin")
    uri2 = fs.uri_for("dir/two.bin")
    fs.put_stream(uri1, io.BytesIO(b"abc"))
    fs.put_stream(uri2, io.BytesIO(b"xyzw"))
    listed = sorted([o.uri for o in fs.list(fs.uri_for("dir"))])
    assert len(listed) == 2


def test_delete(tmp_path: Path) -> None:
    fs = LocalStorage(tmp_path)
    uri = fs.uri_for("d.txt")
    fs.put_bytes(uri, b"x")
    fs.delete(uri)
    assert not fs.exists(uri)
