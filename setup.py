from setuptools import setup

setup(
    name='fiatlux',
    version='0.1.0',    
    description='Fiatlux is a project that aims to offer a versatile environment for multiple optical simulation',
    url='https://github.com/bidou2002/fiatlux',
    author='Pierre Janin-Potiron',
    author_email='pierre.janinpotiron@gmail.com',
    license='MIT License',
    packages=['fiatlux'],
    install_requires=['matplotlib',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
