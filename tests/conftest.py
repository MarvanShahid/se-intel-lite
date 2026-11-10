import subprocess
import textwrap

import pytest

FILES = {
    "pkg/__init__.py": "from .core import Engine\n",
    "pkg/base.py": textwrap.dedent('''
        class Base:
            def run(self):
                return self.step()

            def step(self):
                return 1
        '''),
    "pkg/core.py": textwrap.dedent('''
        from .base import Base
        from . import util


        class Engine(Base):
            def step(self):
                if util.helper(3) > 2:
                    for i in range(3):
                        if i:
                            return i
                return util.helper(1)

            def dispatch_request(self, x):
                return self.run() + x
        '''),
    "pkg/util.py": textwrap.dedent('''
        def helper(n):
            """Return n doubled."""
            return n * 2
        '''),
    "tests/test_core.py": "from pkg.core import Engine\n\ndef test_e():\n    assert Engine().run()\n",
}


@pytest.fixture
def tiny_repo(tmp_path):
    for p, src in FILES.items():
        f = tmp_path / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(src)
    return tmp_path


@pytest.fixture
def git_repo(tiny_repo):
    def git(*a):
        subprocess.run(["git", "-C", str(tiny_repo), *a], check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "T")
    git("add", ".")
    git("commit", "-q", "-m", "initial import")
    (tiny_repo / "pkg/util.py").write_text("def helper(n):\n    return n * 2 + 0\n")
    git("commit", "-qam", "Fix off-by-one bug in helper")
    (tiny_repo / "pkg/core.py").write_text((tiny_repo / "pkg/core.py").read_text() + "\n# note\n")
    git("commit", "-qam", "Fix typo in comment")
    return tiny_repo
