from setuptools import setup, find_packages

setup(
    name='e3nn_Unet',
    version='0.1.0',
    description='Segmentation network with O3 and SO3 equivariant convolutions',
    url='https://github.com/SCAN-NRAD/e3nn_Unet',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='MRI, segmentation',
    package_dir={'': 'e3nn_Unet'},
    packages=find_packages(where='e3nn_Unet'),
    python_requires='>=3.7',
    install_requires=['e3nn==0.3.5',
                      'nnunet==1.6.5',
                      'numpy>=1.20.3',
                      'torch>=1.8.1'],
    license="MIT",
    license_files="LICENSE",
)
