# Neural Programmer

Original project can be found in https://github.com/tensorflow/models, mantained by Arvind Neelakantan (arvind2505)

The original code has been changed:

* Support custom questions
* Support custom tables
* Added "demo" mode (visual and console)
* Support custom descriptions for the columns in the testing target table: improve column selection process
* When the max_entry length for the column names is greater than 1, the embeddings are averaged instead of just added up.
* Fixed multiple bugs on the code
* Cleaned code for debugging reasons

## Original Instructions

Implementation of the Neural Programmer model described in [paper](https://openreview.net/pdf?id=ry2YOrcge)

Download and extract the data from [dropbox](https://www.dropbox.com/s/9tvtcv6lmy51zfw/data.zip?dl=0). Change the ``data_dir FLAG`` to the location of the data.

### Training 
``python neural_programmer.py`` 

The models are written to FLAGS.output_dir

### Testing 
``python neural_programmer.py --evaluator_job=True``

The models are loaded from ``FLAGS.output_dir``. The evaluation is done on development data.

In case of errors because of encoding, add ``"# -*- coding: utf-8 -*-"`` as the first line in ``wiki_data.py``
