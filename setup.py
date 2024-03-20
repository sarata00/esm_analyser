import setuptools

with open("README.md", "r") as f:
    readme = f.read()

sources = {
    "scripts": "scripts",
    "scripts.analysis": "scripts/analysis",
    "scripts.generate_tensor": "scripts/tensor_generators",
    "scripts.streamlit": "scripts/streamlit"

}

setuptools.setup(
    name="Embedding analyser",
    version="1.0",
    description="",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Sara Tolosa Alarcón",
    author_email="sara.tolosa@bsc.es",
    
    package_dir=sources,
    packages=sources.keys(),
    keywords=["esm-2", "embedding", "correlation analysis", "dimensionality reduction"],
    
)