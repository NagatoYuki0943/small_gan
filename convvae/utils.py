from pathlib import Path


def remove_files(dir: Path, prefix: str):
    """根据前缀删除文件"""
    files = dir.glob("*")
    for file in files:
        if file.name.startswith(prefix):
            file.unlink(missing_ok=True)


def test_remove_files():
    remove_files(Path("work_dirs/cifar10"), "best_discriminator")


def save_file(file_path: Path, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    with open(save_dir / file_path.name, "w", encoding="utf-8") as f:
        f.write(text)


def test_save_file():
    save_file(Path(__file__), Path("work_dirs/cifar10"))


if __name__ == "__main__":
    ...
    # test_remove_files()
    test_save_file()
