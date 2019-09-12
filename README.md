# captcha-recognition

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/272198b96453415ebc1416b14f7a8f54)](https://www.codacy.com/manual/saraivaufc/captcha-recognition?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=saraivaufc/captcha-recognition&amp;utm_campaign=Badge_Grade)

### Create and activate a virtual environment:

```
$ virtualenv env -p python3
$ source env/bin/activate
```


### Install TensorFlow-GPU
```
(env) $ pip3 tensorflow-gpu==2.0.0-beta1
```

### Install Others Requirements

```
(env) $ pip3 install -r requirements.txt
```

### To build datasets (data/train.h5 and data/test.h5)
```
(env) $ python3 build_datasets.py
```

### To train cnn
```
(env) $ python3 train.py
```

### To predict captcha image
```
(env) $ python3 predict.py
```
