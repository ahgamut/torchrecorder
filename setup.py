from setuptools import setup, find_packages

VERSION = "1.0.0a"

with open("./README.md") as f:
    long_description = f.read()

setup_info = dict(
    name="torchrec",
    version=VERSION,
    author="Gautham Venkatasubramanian",
    author_email="ahgamut@gmail.com",
    url="https://github.com/ahgamut/pytorchrec",
    description="Record execution graphs of PyTorch neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages("src"),
    package_dir={"": "src"},
    zip_safe=True,
    install_requires=["torch>=1.3", "graphviz"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Visualization",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
    ],
)

setup(**setup_info)
