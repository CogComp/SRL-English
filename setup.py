import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SRL-English-celine1",
    version="0.0.1",
    author="Celine",
    author_email="celine1@example.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CogComp/SRL-English",
    project_urls={
        "Bug Tracker": "https://github.com/CogComp/SRL-English/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    #package_dir={"": "src"},
    package_dir={"": ""},
    # packages=setuptools.find_packages(where="src"),
    packages=setuptools.find_packages(where=""),
    python_requires=">=3.6",
)
