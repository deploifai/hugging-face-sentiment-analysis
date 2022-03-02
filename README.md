# Train a machine learning model using Hugging Face datasets on Deploifai

To read the full post with images, go to the [Deploifai Blog](https://blog.deploif.ai/posts/experiment_hf)

This is an example for using Deploifai experiments to train a machine learning model. The model will be built using TensorFlow (python). To make things interesting, we will also use the `datasets` library to fetch the [tweet_eval](https://huggingface.co/datasets/tweet_eval) dataset from Hugging Face. If you want to directly head over to the code, find this example and others at [GitHub](https://github.com/deploifai/hugging-face-sentiment-analysis).

## Setting up the project locally

I like to set up a virtual environment first. Using `virtualenv` I can create an environment where I will install all the dependencies. Then I install the different dependencies. In case of this project we are going to need tensorflow, datasets and transformers.

```bash
pip install tensorflow datasets transformers
```

Once these packages are installed, we are good to start writing in our script for training.

At this point, I should warn you that I am going to assume that you have some understanding of the concepts and jargon that comes with TensorFlow, NLP and Hugging Face. This means that you should at least Google those things if you find yourself lost while reading.

### Getting datasets from Hugging Face

We have decided to use the `datasets` pip package from Hugging Face for this. Let’s see how to simply get the dataset.

```python
from datasets import load_dataset

# Load the train split of the emoji sub-dataset of tweet_eval
train_dataset = load_dataset("tweet_eval", "emoji", split='train')
# Load the test split of the emoji sub-dataset of tweet_eval
test_dataset = load_dataset("tweet_eval", "emoji", split='test')
```

The `load_dataset` function is handy and can be used simply to make the import.

Since we know we are going to use these datasets to train a TensorFlow model, we must pre-process this dataset and format it correctly so that we can feed it to `model.fit()` method.

To format the dataset, we are going to make a function:

```python
import numpy as np
from transformers import RobertaTokenizer
from keras.preprocessing.sequence import pad_sequences

def process_dataset(dataset):
  tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
  sentences = [d.as_py() for d in dataset.data['text']]
  labels = np.array([l.as_py() for l in dataset.data['label']])
  sequences = [tokenizer(s)['input_ids'] for s in sentences]
  padded = pad_sequences(sequences=sequences, padding='post', maxlen=35)
  return padded, labels
```

NLP models require you to use tokenizers to “tokenize” words. They help us represent sentences as vectors. If you wish to learn this concept, there are wonderful resources on YouTube to do so!

We need to return a padded sequence that represents each of the sentences (tweets), along with the labels (which indicate the sentiment in emoji) as `numpy` arrays that we can give to the `model.fit()` method to train the model.

We can process the two datasets that we imported earlier using this function. The code we have now looks something like this:

```python
import numpy as np
from datasets import load_dataset
from keras.preprocessing.sequence import pad_sequences
from transformers import RobertaTokenizer

def process_dataset(dataset):
  tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
  sentences = [d.as_py() for d in dataset.data['text']]
  labels = np.array([l.as_py() for l in dataset.data['label']])
  sequences = [tokenizer(s)['input_ids'] for s in sentences]
  padded = pad_sequences(sequences=sequences, padding='post', maxlen=35)
  return padded, labels

train_dataset = load_dataset("tweet_eval", "emoji", split='train')
test_dataset = load_dataset("tweet_eval", "emoji", split='test')

training_padded, training_labels = process_dataset(train_dataset)
test_padded, test_labels = process_dataset(test_dataset)
```

Our datasets are ready and now we can build the model itself. This is where I give you another disclaimer. This model is not going to perform well. Not only can this model be improved, there are other models out there that will perform much better for NLP tasks. The only reason we have this here, is to provide a basic structure from which one can possibly build greater things.

```python
from keras import Sequential, Model
from keras.layers import Embedding, Flatten, GlobalAveragePooling1D, Dense, Bidirectional, LSTM

model = Sequential([
  Embedding(50265, 64, input_length=35), # Roberta tokeniser vocabulary size is 50,265
  Bidirectional(LSTM(64, return_sequences=True)),
  Bidirectional(LSTM(32)),
  Dense(512, activation='relu'),
  Dense(20, activation='softmax') # The emoji dataset has 20 classes
])

model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()

"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 embedding (Embedding)       (None, 35, 64)            3216960

 bidirectional (Bidirectiona  (None, 35, 128)          66048
 l)

 bidirectional_1 (Bidirectio  (None, 64)               41216
 nal)

 dense (Dense)               (None, 512)               33280

 dense_1 (Dense)             (None, 20)                10260

=================================================================
Total params: 3,367,764
Trainable params: 3,367,764
Non-trainable params: 0
_________________________________________________________________
"""
```

This model can now be trained! And it is simple to do so:

```python
model.fit(training_padded, training_labels, epochs=100, validation_data=(test_padded, test_labels))
model.save("model")
```

## Setting up training on Deploifai

Let’s clarify why we are doing this step. I have a laptop that cannot handle this workload, even though this is a small training task. The requirements for training machine learning models only becomes harder from here. However, me and my team use cloud services to offload these tasks. I have to spin up a VM on the cloud, set up the project dependencies and run the training.

Last generation stuff with Deploifai! It is possible to automate all the steps above and simply run a training on Deploifai from a GitHub repo.

### Sign up on the web

If you have not already, sign up on Deploifai: [https://deploif.ai/signup](https://deploif.ai/signup).

### Connect your cloud service

Then simply get started by creating a project from the dashboard. Give the project a name and connect a cloud service account that you can use to spin up VMs. We will take care of the cloud service side of things once you have set it up.

Learn to set up the cloud service on Deploifai: [https://docs.deploif.ai/cloud-services/connect-your-account](https://docs.deploif.ai/cloud-services/connect-your-account)

### Connect your GitHub repository

Next, we connect our GitHub repo in the project settings. Go to **Project > Settings** to find the GitHub repository integration setting.

Once the repository is connected, we are ready to start running the experiment!

### Set up the experiment

The key configurations while setting up experiments are Artifacts and Start scripts.

#### Artifacts

We specify the `model` folder as our artifacts directory since that is where we output the trained model.

#### Start scripts

Next is the start scripts. In my case, I can leave this blank since I will put the `requirements.txt` in the root directory and also put a shell script called `experiment.sh` in the root as well. If your scenario calls for other names, you can put the path in this configuration assuming relative path to the root directory of your project.

#### A dummy dataset

This last note is to point out the current limitation of Deploifai’s API. At the moment, we require experiments to have datasets attached to them. While this dataset is going to be empty, we must still make one so that we can finish creating the experiment. This neither costs money on any of the cloud services, nor causes any issues in our experiment since we are not using any datasets imported by Deploifai.

#### Run the experiment, download the model after training

Deploifai takes it from here to create an environment on your cloud service where this model will be trained. We wait for Deploifai to finish deploying the resources.

Once the experiment is created, we can run these experiments and see the logs. More importantly, once the experiment is finished, the artifacts will be exported and available for download!

Use this trained model for your application! Of course, this model is not nearly good enough to run on any application, but you can follow the same workflow to build more machine learning models.

Please let us know if you followed this guide via [Twitter](https://twitter.com/deploifai). If you found issues in this post, please send us an email at [contact@deploif.ai](mailto:contact@deploif.ai). Thank you for reading!
