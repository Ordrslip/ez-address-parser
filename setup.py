from os import path
from setuptools import setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md")) as f:
    long_description = f.read()

setup(
    name="ez-address-parser",
    packages=["ez_address_parser"],
    entry_points={"console_scripts": []},
    include_package_data=True,
    package_data={},
    install_requires=[],
    setup_requires=["setuptools_scm", "pytest-runner"],
    use_scm_version=True,
    tests_require=["pytest"],
    test_suite="tests",
    author="Zeheng li",
    author_email="imzehengl@gmail.com",
    maintainer="Zeheng li",
    maintainer_email="imzehengl@gmail.com",
    description="An address parser",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zehengl/ez-address-parser",
)
