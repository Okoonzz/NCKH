import os
import re
import shutil
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

ZIP_ROOT = Path(r"MarauderMap-20251109T161842Z-1-002\MarauderMap")  
OUT_DIR  = Path(r"getdata\binary") 
PASSWORD = "maraudermap"

TARGET_DIR_NAME = "230701-Win32-EXE-all-7802"


def extract_zip(zip_path: Path, dest: Path, password: str):
    pw = password.encode("utf-8") if password else None

    try:
        import pyzipper  
        with pyzipper.AESZipFile(zip_path) as zf:
            if pw:
                zf.pwd = pw
            zf.extractall(dest)
        return
    except ModuleNotFoundError:
        pass  
    except Exception as e:
        print(f"[warn] pyzipper lỗi với {zip_path.name}: {e}. Thử zipfile tiêu chuẩn...")

    with zipfile.ZipFile(zip_path) as zf:
        if pw:
            zf.setpassword(pw)
        zf.extractall(dest)


def find_payload(extract_root: Path, hash_name: str) -> Path | None:
    candidates = list(extract_root.rglob(TARGET_DIR_NAME))
    search_roots = candidates if candidates else [extract_root]

    for root in search_roots:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            name_ok  = (p.name == hash_name)
            stem_ok  = (p.stem == hash_name)
            if name_ok or stem_ok:
                return p

    for root in search_roots:
        for p in root.rglob(f"*{hash_name}*"):
            if p.is_file():
                return p

    for root in search_roots:
        files = [p for p in root.rglob("*") if p.is_file()]
        if len(files) == 1:
            return files[0]

    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    zips = sorted(ZIP_ROOT.glob("*.zip"))
    if not zips:
        print(f"[info] Không tìm thấy *.zip trong: {ZIP_ROOT}")
        return

    ok, miss = 0, 0
    for zp in zips:
        hash_name = zp.stem
        print(f"\n[>] Xử lý {zp.name}")

        with TemporaryDirectory(prefix=f"ext_{hash_name}_") as td:
            tdir = Path(td)
            try:
                extract_zip(zp, tdir, PASSWORD)
            except Exception as e:
                print(f"[err] Giải nén thất bại {zp.name}: {e}")
                miss += 1
                continue

            payload = find_payload(tdir, hash_name)
            if not payload:
                print(f"[warn] Không tìm thấy file có tên khớp hash trong {zp.name}")
                miss += 1
                continue

            dest = OUT_DIR / payload.name
            if dest.exists():
                i = 1
                while True:
                    candidate = OUT_DIR / f"{payload.stem}_{i}{payload.suffix}"
                    if not candidate.exists():
                        dest = candidate
                        break
                    i += 1

            shutil.copy2(payload, dest)
            print(f"[ok] Copied -> {dest}")
            ok += 1

    print(f"\n[done] Thành công: {ok}, Không tìm thấy/có lỗi: {miss}")


if __name__ == "__main__":
    main()