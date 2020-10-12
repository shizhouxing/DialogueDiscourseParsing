# Dialogue Discourse Parsing

Code for our paper:

Zhouxing Shi and Minlie Huang. A Deep Sequential Model for Discourse Parsing on Multi-Party Dialogues. In AAAI, 2019.

```
@inproceedings{shi2019deep,
  title={A Deep Sequential Model for Discourse Parsing on Multi-Party Dialogues},
  author={Shi, Zhouxing and Huang, Minlie},
  booktitle={AAAI},
  year={2019}
}
```

## Requirements

* Python 2.7
* Tensorflow 1.3

## Data Format

We use JSON format for data. The data file should contain an array consisting of examples represented as objects. The format for each example object looks like:

```
{
    // a list of EDUs in the dialogue
    "edus": [ 
        {
            "text": "text of edu 1",
            "speaker": "speaker of edu 1"
        },    
        // ...
    ],
    // a list of relations
    "relations": [
        {
            "x": 0,
            "y": 1,
            "type": "type 1"
        },
        // ...
    ]
}
```

## STAC Corpus

We used the linguistic-only [STAC corpus](https://www.irit.fr/STAC/corpus.html). The latest available verison on the website is [stac-linguistic-2018-05-04.zip](https://www.irit.fr/STAC/stac-linguistic-2018-05-04.zip). It appears that test data is missing in this version. We share the [test data we used](https://drive.google.com/file/d/1KmTw_DbJawNfl6asqtRnG_f4BwXtl4b7/view?usp=sharing) from the 2018-03-21 version.


To process a raw dataset into our JSON format, run:

```
python data_pre.py <input_dir> <output_json_file>
```

There are 1086 dialogues in the training data and 111 dialogues in the test data. This version of dataset and the processing script produce data that are slightly different from those used for our publication. 

On the website of STAC, there is a latest version in python pandas dataframes format, but it appears to be more different from the version mentioned above and it is yet unclear to us how this dataset should be processed.

## Word Vectors

For pre-trained word verctors, we used [GloVe](https://nlp.stanford.edu/projects/glove/) (100d).

## How to Run

```
python main.py {--[option1]=[value1] --[option2]=[value2] ... }
```

Available options can be found at the top of `main.py`.

For example, to train the model with default settings:

```
python main.py --train
```

## Baselines

For baselines (`Deep+MST`, `Deep+ILP`, `Deep+Greedy`), see [here](./baseline).