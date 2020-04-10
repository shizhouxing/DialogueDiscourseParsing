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

We used the [STAC corpus](https://www.irit.fr/STAC/corpus.html). However, it seems that their dataset has been significantly updated after our publication, but we do not have a script to process it for now. Please convert any data you use to the format as showed above before running our code.

For pre-trained word verctors, we used [GloVe](https://nlp.stanford.edu/projects/glove/) (100d).

## How to Run

```
python main.py {--[option1]=[value1] --[option2]=[value2] ... }
```

Available options can be found at the top of `main.py`.

For example, to train the model with default settings:

```
python main.py --is_train
```

