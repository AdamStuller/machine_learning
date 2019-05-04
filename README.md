#Machine learning

This Repo contains various types of machine learning models for classification of hand written digits. 

I contains Neural Network, Decission Tree and Random Forest classifiers. All of them have their own source files and 
configs to modify behaviour of respective files. 

To run the training and subsequently the testing of model run in terminal:
```
git clone git@github.com:AdamStuller/machine_learning.git
cd ./machine_learning
```

Once you have cloned the directory and you are inside, you have to unzip the datasets and create folders for them 
and classifiers.
```
unzip ./datasets.zip
mkdir ./data/classifiers
```

Then start virtual environment and install everything you need 
```
python3.7 -m venv ./venv
./venv/bin/pip3.7 install -r ./requirements.txt
```

Now you can specify behaviour in every config and run each model by calling for instance:
```
./venv/bin/python3.7 ./decision_tree.py
```