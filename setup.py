from pathlib import Path
from typing import List

import setuptools

# The directory containing this file
HERE = Path(__file__).parent.resolve()

# The text of the README file
NAME = 'fedot_ind'
VERSION = '0.5.0'
AUTHOR = 'NSS Lab'
AUTHOR_EMAIL = 'itmo.nss.team@gmail.com'
SHORT_DESCRIPTION = 'Automated machine learning framework for time series analysis'
LONG_DESC_TYPE = 'text/x-rst'
README = Path(HERE, 'README_en.rst').read_text(encoding='utf-8')
EXCLUDED_PACKAGES = ['test*', 'examples', 'benchmark']
URL = 'https://github.com/aimclub/Fedot.Industrial'
REQUIRES_PYTHON = '>=3.8'
LICENSE = 'BSD 3-Clause'
KEYWORDS = 'automated machine learning, time series analysis, anomaly detection, classification'


def _readlines(*names: str, **kwargs) -> List[str]:
    encoding = kwargs.get('encoding', 'utf-8')
    lines = Path(__file__).parent.joinpath(
        *names).read_text(encoding=encoding).splitlines()
    return list(map(str.strip, lines))


def _extract_requirements(file_name: str):
    return [line for line in _readlines(
        file_name) if line and not line.startswith('#')]


def _get_requirements(req_name: str):
    requirements = _extract_requirements(req_name)
    return requirements


setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=SHORT_DESCRIPTION,
    long_description=README,
    long_description_content_type=LONG_DESC_TYPE,
    url=URL,
    python_requires=REQUIRES_PYTHON,
    license=LICENSE,
    packages=setuptools.find_packages(exclude=EXCLUDED_PACKAGES),
    include_package_data=True,
    install_requires=_get_requirements('requirements.txt'),
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],
    keywords=KEYWORDS
)
