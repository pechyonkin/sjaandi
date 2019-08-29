import setuptools

with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    name='sjaandi',
    version='0.1.6',
    scripts=None,
    author='Max Pechyonkin',
    author_email='maxim.pechyonkin@gmail.com',
    description='Library for visual similarity search and visualization',
    # long_description=long_description,
    # long_description_content_type='text/markdown',
    url='https://github.com/pechyonkin/sjaandi',
    packages=setuptools.find_packages(),
    install_requires=[
        'torch>=1.2.0',
        'torchvision>=0.4.0',
        'fastai>=1.0.57',
        'matplotlib',
        'numpy',
        'scikit-learn',
        'python-dotenv',
        'click',
        'rasterfairy-py3',
    ],
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
