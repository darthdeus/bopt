import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="bopt",
    version="0.0.1a0",
    author="Jakub Arnold",
    author_email="darthdeus@gmail.com",
    description="Bayesian Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/darthdeus/bopt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
