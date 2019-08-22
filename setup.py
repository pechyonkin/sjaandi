import setuptools

with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    name='sjaandi',
    version='0.1',
    scripts=None,
    author='Max Pechyonkin',
    description='Library for visual similarity search and visualization',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/pechyonkin/sjaandi',
    packages=setuptools.find_packages(),
    install_requires=[
        'fastai',
        'matplotlib',
        'numpy',
        'scikit-learn',
        'rasterfairy @ git+git://github.com/pechyonkin/RasterFairy@python3',
    ],
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
