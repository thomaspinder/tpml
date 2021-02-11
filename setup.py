from setuptools import setup, find_packages


def parse_requirements_file(filename):
    with open(filename, encoding="utf-8") as fid:
        requires = [l.strip() for l in fid.readlines() if l]
    return requires

setup(
    name="tpml",
    version="0.1.0",
    author="Thomas Pinder",
    author_email="t.pinder2@lancaster.ac.uk",
    packages=find_packages(".", exclude=["tests"]),
    license="LICENSE",
    description="Machine learning helper funcs.",
    install_requires=parse_requirements_file("requirements.txt"),
    keywords=["machine-learning"],
)