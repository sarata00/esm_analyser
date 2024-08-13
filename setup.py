import setuptools

with open("README.md", "r") as f:
    readme = f.read()

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
    entry_points={
        "console_scripts": [
            "embedding_analyzer = scripts.analysis.__main__:main",
            "embedding_generator_esm = scripts.embedding_generators.embedding_generator_esm:main",
            "embedding_generator_HF = scripts.embedding_generators.embedding_generator_hugging_face:main",
        ]
    }, 
)