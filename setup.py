import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='overlay_viz',
    version='0.1',
    scripts=['bin/overlay_viz'],
    author="Jon Muhovic",
    author_email="jon.muhovic@fe.uni-lj.si",
    description="Overlay Visualization Toll",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vicoslab/dl_visualizer",
    packages=setuptools.find_packages(),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
