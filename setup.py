from setuptools import setup

VERSION = "0.1.0"

long_description = ""

setup_info = dict(
    name="torchrec",
    version=VERSION,
    author="Gautham Venkatasubramanian",
    author_email="ahgamut@gmail.com",
    url="https://github.com/ahgamut/pytorchrec",
    description="A small package to record execution graphs of PyTorch neural networks",
    long_description=long_description,
    license="MIT",
    package_dir={"": "src"},
    zip_safe=True,
    install_requires=["torch>=1.3", "graphviz"],
)

setup(**setup_info)
