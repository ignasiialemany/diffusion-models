# Documentation

## Notebooks

To view the [Jupyter Notebooks](https://jupyter.org/), you need to install `jupyter`.
In the INSTALL guide, we recommend using conda.
Simply execute `conda install notebook` and then start the server with `jupyter-notebook`.

To compile multiple notebooks into a single document, we can use [`nbmerge`](https://github.com/jbn/nbmerge).
This can be installed through `pip install nbmerge`.
It is then possible to execute the following snippet to combine notebooks and compile them to a PDF.

```bash
title='Code Documentation' # will be filename and title
nbmerge *.ipynb | \
jupyter-nbconvert --stdin \
                  --to pdf --output "${title}.pdf" \
                  --execute --ExecutePreprocessor.kernel_name=python
```

If the code cells are not desired, add the `--no-input` flag to `jupyter-nbconvert`.
