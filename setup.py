import os
from setuptools import setup

# For guidance on setuptools best practices visit
# https://packaging.python.org/guides/distributing-packages-using-setuptools/
project_name = os.getcwd().split("/")[-1]
version = "0.1.0"
package_description = "<Provide short description of package>"
url = "https://github.com/ai2es/" + project_name
# Classifiers listed at https://pypi.org/classifiers/
classifiers = ["Programming Language :: Python :: 3"]
setup(name="evidential",  # Change
      version=version,
      description=package_description,
      url=url,
      author="John Schreck, David John Gagne, Charlie Becker, Gabrielle Gantos",
      license="CC0 1.0",
      classifiers=classifiers,
      packages=["evml", "evml.keras", "evml.torch"])
