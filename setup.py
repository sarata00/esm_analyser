import setuptools

with open("README.md", "r") as f:
    readme = f.read()

with open("requirements.txt", "r") as f:
    install_requirements = f.read().splitlines()

setuptools.setup(
    name="Embedding analyzer",
    version="1.0",
    description="",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Sara Tolosa Alarcón",
    author_email="sara.tolosa@bsc.es",
    
    packages=setuptools.find_packages(exclude=["tests"]),
    keywords=["esm-2", "embedding", "correlation analysis", "dimensionality reduction"],
    python_requires=">3.6",
    install_requirements=install_requirements,
    entry_points={
        "console_scripts": [
            "embedding_analyzer=scripts.analysis.__main__:main",
        ]
    }, 
)