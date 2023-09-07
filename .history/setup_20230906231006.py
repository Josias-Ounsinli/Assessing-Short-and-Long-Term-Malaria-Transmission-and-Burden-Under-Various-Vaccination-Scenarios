""" Setup file """

from setuptools import setup, find_packages


with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open("requirements.txt", encoding="utf-8") as req:
    req_list = req.readlines()
    req_list = [sd.replace("\n", "") for sd in req_list]

requirements = req_list
test_requirements = ["pytest>=3"]

setup(
    author="Josias-Ounsinli Master Thesis",
    email="josias.ouns.djossou@gmail.com",
    description="Assessing-the-effects-of-various-vaccination-scenarios-on-malaria-dynamics",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Beta",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    keywords="scripts",
    name="scripts",
    packages=find_packages(
        include=[
            "scripts",
            "scripts.*",
            "data",
            "data.*",
            "tests",
            "tests.*",
            "notebooks",
        ]
    ),
    test_suite="tests",
    tests_require=test_requirements,
    url="",
    version="1.0.0",
    zip_safe=False,
)
