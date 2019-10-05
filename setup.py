from distutils.core import setup

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

def setup_package():
    metadata = dict(name='deepstack',
                    maintainer='Julio Borges',
                    long_description=LONG_DESCRIPTION,
                    long_description_content_type='text/markdown',
                    description='DeepStack: Ensembling Keras Deep Learning Models into the next Performance Level',
                    license='MIT',
                    url='https://github.com/jcborges/DeepStack',
                    version='0.0.3',
                    install_requires=[
                        'numpy>=1.16.4',
                        'keras>=2.2.5',
                        'tensorflow>=1.14.0',
                        'scikit-learn>=0.21.2'
                    ],
                    )

    setup(**metadata)


if __name__ == "__main__":
    setup_package()