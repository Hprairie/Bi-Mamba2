# Copyright (c) 2024, Hayden Prairie.

from glob import glob

pytest_plugins = [
    fixture_file.replace("/", ".").replace(".py", "").split(".", 1)[1]
    for fixture_file in glob("tests/fixtures/[!__]*.py", recursive=True)
]
