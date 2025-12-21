import os
import shutil
import subprocess
from pathlib import Path

def run_hafs(args, cwd=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.getcwd()}/src:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        ["python3", "-m", "hafs.cli"] + args,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True
    )
    return result

def test_cli():
    test_dir = Path("/tmp/hafs_cli_test")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)
    
    print("Testing 'hafs init'...")
    res = run_hafs(["init"], cwd=test_dir)
    print(res.stdout)
    assert (test_dir / ".context").is_dir()
    
    source_dir = test_dir / "my_docs"
    source_dir.mkdir()
    (source_dir / "note.md").write_text("hello")
    
    print("Testing 'hafs mount'...")
    res = run_hafs(["mount", "knowledge", str(source_dir)], cwd=test_dir)
    print(res.stdout)
    assert (test_dir / ".context" / "knowledge" / "my_docs").is_symlink()
    
    print("Testing 'hafs list'...")
    res = run_hafs(["list"], cwd=test_dir)
    print(res.stdout)
    assert "knowledge" in res.stdout.lower()
    assert "my_docs" in res.stdout
    
    print("Testing 'hafs unmount'...")
    res = run_hafs(["unmount", "knowledge", "my_docs"], cwd=test_dir)
    print(res.stdout)
    assert not (test_dir / ".context" / "knowledge" / "my_docs").exists()

    # Test project discovery from subdirectory
    sub_dir = test_dir / "subdir"
    sub_dir.mkdir()
    print("Testing project discovery from subdirectory...")
    res = run_hafs(["list"], cwd=sub_dir)
    assert "Context Root" in res.stdout
    assert str(test_dir / ".context") in res.stdout
    print("Discovery successful!")

if __name__ == "__main__":
    try:
        test_cli()
        print("\nALL CLI TESTS PASSED")
    except Exception as e:
        import traceback
        print(f"\nTEST FAILED: {e}")
        traceback.print_exc()
        exit(1)
