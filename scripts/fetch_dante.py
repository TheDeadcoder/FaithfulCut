import argparse
import io
import posixpath
import struct
import urllib.request
import zipfile
import zlib
from pathlib import Path

URL = "https://zenodo.org/api/records/18900675/files/ML_TCSP.zip/content"
ROOT = "ML_TCSP/"


class RangeFile(io.RawIOBase):
    """A read-only view of a remote file that fetches only the byte ranges read."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        data = fetch(self.url, self.pos, self.pos + n)
        self.pos += len(data)
        return data


def fetch(url: str, start: int, end: int) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"}),
                                timeout=600) as r:
        data = r.read()
    if r.status != 206 or len(data) != end - start:
        raise IOError(f"range {start}-{end} not served")
    return data


def wanted(name: str) -> bool:
    rel = name[len(ROOT):]
    if not name.startswith(ROOT) or name.endswith("/") or "/." in name or "__pycache__" in name:
        return False
    source = rel.startswith("src/") and "tcp_features" not in rel
    return "/" not in rel or rel.startswith("experiments/RQ1/") or source


def main():
    parser = argparse.ArgumentParser(description="Fetch DANTE's code, paper, published RQ1 results and sample "
                                                 "processed builds from its replication package")
    parser.add_argument("out", help="a folder to create")
    out = Path(parser.parse_args().out)
    out.mkdir(parents=True)
    archive = zipfile.ZipFile(RangeFile(URL))
    infos = sorted(archive.infolist(), key=lambda i: i.header_offset)
    ends = [i.header_offset for i in infos[1:]] + [archive.start_dir]
    spans = []
    for info, end in zip(infos, ends):
        if not wanted(info.filename):
            continue
        if spans and info.header_offset - spans[-1][1] < 2_000_000:
            spans[-1][1] = end
            spans[-1][2].append(info)
        else:
            spans.append([info.header_offset, end, [info]])
    root, written = out.resolve(), 0
    for start, end, members in spans:
        blob = fetch(URL, start, end)
        for info in members:
            h = info.header_offset - start
            name_len, extra_len = struct.unpack("<HH", blob[h + 26:h + 30])
            data = blob[h + 30 + name_len + extra_len:h + 30 + name_len + extra_len + info.compress_size]
            raw = zlib.decompressobj(-15).decompress(data) if info.compress_type == zipfile.ZIP_DEFLATED else data
            if len(raw) != info.file_size or zlib.crc32(raw) != info.CRC:
                raise IOError(f"{info.filename} failed its checksum")
            target = (out / posixpath.normpath(info.filename[len(ROOT):])).resolve()
            if not str(target).startswith(str(root) + "/"):
                raise IOError(f"{info.filename} points outside {out}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            written += 1
    print(f"{written} files written to {out}")


if __name__ == "__main__":
    main()
