from setuptools import find_packages, setup

setup(
    name="cluster_optimizer",
    author="nick_gigliotti",
    author_email="ndgigliotti@gmail.com",
    description="A GridSearchCV-like hyperparameter optimizer for clustering (no cross-validation).",
    license="MIT",
    version='0.0.1',
    url="https://github.com/ndgigliotti/cluster-optimizer",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.0.1",
        "numpy>=1.19.5"
    ],
)