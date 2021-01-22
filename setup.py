from distutils.core import setup

setup(name='diffusion-models',
      version='0.2',
      description='Diffusion Models',
      author='Jan Niklas Rose',
      author_email='janniklas.rose@gmail.com',
      url='https://github.com/janniklasrose/diffusion-models',
      packages=['diffusion', 'diffusion.analytical', 'diffusion.mcrw'],
      install_requires=[
            'numpy',
            'scipy',
            'matplotlib',
            'property-cached',
            'chebpy@git+https://github.com/chebpy/chebpy.git',  # get the right chebpy
      ]
     )
