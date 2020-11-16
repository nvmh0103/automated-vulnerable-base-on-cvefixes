# -*- coding: utf-8 -*-

import argparse
import sys
import logging
from logging.handlers import RotatingFileHandler
import tensorflow as tf
import pandas as pd
import h5py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

import SVD_common as svdc

logging.debug("Tensorlfow version: ", tf.__version__)
logging.debug("Eager mode: ", tf.executing_eagerly())
logging.debug("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

def input_arguments():
    """
    Takes the input from terminal and returns a parse dictionary for arguments
    
    Parameters
    ----------
    NONE
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d" ,"--dataset", type= str, nargs= 1,
                        help="dataset that will be vectorized, eg: VDISC")
    parser.add_argument("-p" ,"--pickle",
                        help="pickle dataset files", action="store_const", const=True)
    parser.add_argument("-t" ,"--train",
                        help="train model", action="store_const", const=True)
    parser.add_argument("-ll", "--loglevel", type= int, nargs = '?', const= 0, default= 0,
                        help="set logging level. Default = 0 (NOTSET)."
                        " Treshold levels: "
                        " DEBUG = 10," 
                        " INFO = 20,"
                        " WARNING = 30,"
                        " ERROR = 40,"
                        " CRITICAL = 50."
                        )

    parser.add_argument("-c", "--console", type= int, nargs = '?', const= 1, default= 0,
                        help="log file output type, default (or 1) returns .log file. '-c' returns terminal view")
    global args
    args = vars(parser.parse_args())
    if(args["dataset"] == None):
        parser.print_help()
        sys.exit()
    return args
    
def trainModel():
    global args;
    # Generate random seed
    #myrand=np.random.randint(1, 99999 + 1)
    myrand=71926
    np.random.seed(myrand)
    tf.random.set_seed(myrand)
    logging.debug("Random seed is: %s",myrand)
    
    # Set the global value
    WORDS_SIZE=10000
    INPUT_SIZE=500
    NUM_CLASSES=2
    MODEL_NUM=0
    EPOCHS=2
    
    train=pd.read_pickle("pickle_file/"+args['dataset'][0]+"_train.pickle")
    validate=pd.read_pickle("pickle_file/"+args['dataset'][0]+"_validate.pickle")
    test=pd.read_pickle("pickle_file/"+args['dataset'][0]+"_test.pickle")
    
    for dataset in [train, validate, test]:
        for index, row in dataset.iterrows():
            dataset.at[index, 'combine'] = (row[1] == True or row[2] == True or row[3] == True or row[4] == True or row[5] == True)
        
    for dataset in [train, validate, test]:
        dataset.iloc[:,6] = dataset.iloc[:,6].map({False: 0, True: 1})
            
    x_all = train['functionSource']
    one = train[train.iloc[:,1]==1].index.values.astype(int)
    zero = train[train.iloc[:,1]==0].index.values.astype(int)
    # Tokenizer with word-level
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=False)
    tokenizer.fit_on_texts(list(x_all))
    del(x_all)
    logging.debug('Number of tokens: %s',len(tokenizer.word_counts))
    # Reducing to top N words
    tokenizer.num_words = WORDS_SIZE
    # Top 10 words
    sorted(tokenizer.word_counts.items(), key=lambda x:x[1], reverse=True)[0:10]
    ## Tokkenizing train data and create matrix
    list_tokenized_train = tokenizer.texts_to_sequences(train['functionSource'])
    x_train = tf.keras.preprocessing.sequence.pad_sequences(list_tokenized_train, 
                                      maxlen=INPUT_SIZE,
                                      padding='post')
    x_train = x_train.astype(np.int64)
    
    ## Tokkenizing test data and create matrix
    list_tokenized_test = tokenizer.texts_to_sequences(test['functionSource'])
    x_test = tf.keras.preprocessing.sequence.pad_sequences(list_tokenized_test, 
                                     maxlen=INPUT_SIZE,
                                     padding='post')
    x_test = x_test.astype(np.int64)
    
    ## Tokkenizing validate data and create matrix
    list_tokenized_validate = tokenizer.texts_to_sequences(validate['functionSource'])
    x_validate = tf.keras.preprocessing.sequence.pad_sequences(list_tokenized_validate, 
                                     maxlen=INPUT_SIZE,
                                     padding='post')
    x_validate = x_validate.astype(np.int64)
    
    y_train=[]
    y_test=[]
    y_validate=[]
    
    for col in range(1,6):
        y_train.append(tf.keras.utils.to_categorical(train.iloc[:,col], num_classes=NUM_CLASSES).astype(np.int64))
        y_test.append(tf.keras.utils.to_categorical(test.iloc[:,col], num_classes=NUM_CLASSES).astype(np.int64))
        y_validate.append(tf.keras.utils.to_categorical(validate.iloc[:,col], num_classes=NUM_CLASSES).astype(np.int64))
    
    
    # Create a random weights matrix
    
    random_weights = np.random.normal(size=(WORDS_SIZE, 13),scale=0.01)
    
    # Must use non-sequential model building to create branches in the output layer
    model = tf.keras.Sequential(name="CNN")
    model.add(tf.keras.layers.Embedding(input_dim = WORDS_SIZE,
                                        output_dim = 13,
                                        weights=[random_weights],
                                        input_length = INPUT_SIZE))
    #model.add(tf.keras.layers.GaussianNoise(stddev=0.01))
    model.add(tf.keras.layers.Convolution1D(filters=512, kernel_size=(9), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    # Define custom optimizers
    adam = tf.keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1, decay=0.0, amsgrad=False)
    
    ## Compile model with metrics
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    print("CNN model built: ")
    model.summary()
    
    ## Create TensorBoard callbacks
    
    callbackdir= 'cb'
    
    tbCallback = tf.keras.callbacks.TensorBoard(log_dir=callbackdir, 
                             histogram_freq=1,
                             embeddings_freq=1,
                             write_graph=True, 
                             write_images=True)
    
    tbCallback.set_model(model)
    mld = 'model/model-ALL-last.hdf5'
    
    ## Create best model callback
    mcp = tf.keras.callbacks.ModelCheckpoint(filepath=mld, 
                                             monitor="val_loss",
                                             save_best_only=True, 
                                             mode='auto', 
                                             save_freq='epoch', 
                                             verbose=1)
    
    class_weights = {0: 1., 1: 3.}
    
    history = model.fit(x = x_train,
          y = train.iloc[:,6].to_numpy(),
          validation_data = (x_validate, validate.iloc[:,6].to_numpy()),
          epochs = EPOCHS,
          batch_size = 128,
          verbose =2,
          class_weight= class_weights,
          callbacks=[mcp,tbCallback])
    
    
    #with open('history/History-ALL-40EP-CNN', 'wb') as file_pi:
    #    pickle.dump(history.history, file_pi)
    
    # Load model
    model = tf.keras.models.load_model("model/model-ALL-last.hdf5")
    
    ceiling = 7
    
    results = model.evaluate(x_test, test.iloc[:,6].to_numpy(), batch_size=128)
    for num in range(0,len(model.metrics_names)):
        print(model.metrics_names[num]+': '+str(results[num]))
    
    predicted = model.predict_classes(x_test)
    predicted_prob = model.predict(x_test)
    
    confusion = sklearn.metrics.confusion_matrix(y_true=test.iloc[:,6].to_numpy(), y_pred=predicted)
    print(confusion)
    
    tn, fp, fn, tp = confusion.ravel()
    print('\nTP:',tp)
    print('FP:',fp)
    print('TN:',tn)
    print('FN:',fn)
    
    ## Performance measure
    print('\nAccuracy: '+ str(sklearn.metrics.accuracy_score(y_true=test.iloc[:,6].to_numpy(), y_pred=predicted)))
    print('Precision: '+ str(sklearn.metrics.precision_score(y_true=test.iloc[:,6].to_numpy(), y_pred=predicted)))
    print('Recall: '+ str(sklearn.metrics.recall_score(y_true=test.iloc[:,6].to_numpy(), y_pred=predicted)))
    print('F-measure: '+ str(sklearn.metrics.f1_score(y_true=test.iloc[:,6].to_numpy(), y_pred=predicted)))
    print('Precision-Recall AUC: '+ str(sklearn.metrics.average_precision_score(y_true=test.iloc[:,6].to_numpy(), y_score=predicted_prob)))
    print('AUC: '+ str(sklearn.metrics.roc_auc_score(y_true=test.iloc[:,6].to_numpy(), y_score=predicted_prob)))
    print('MCC: '+ str(sklearn.metrics.matthews_corrcoef(y_true=test.iloc[:,6].to_numpy(), y_pred=predicted)))
    
    model.metrics_names
    
    epochs_range = range(len(history.history[model.metrics_names[1]]))


    plt.figure(figsize=(10,5))
    plt.plot(epochs_range, history.history[(model.metrics_names[1])], 'b', label='Accuracy', color='red')
    plt.plot(epochs_range, history.history['val_%s'%(model.metrics_names[1])], 'b', label='Val accuracy', color='green')
    plt.title('Training & validation accuracy (Result 4)')
    plt.legend()
    
    plt.figure(figsize=(10,5))
    plt.plot(epochs_range, history.history[model.metrics_names[0]], 'b', label='Loss', color='red')
    plt.plot(epochs_range, history.history['val_%s'%model.metrics_names[0]], 'b', label='Val loss', color='green')
    plt.title('Training & validation loss (Result 4)')
    plt.legend()
    
    
    

def main(): 
    global args
    args = input_arguments()
    global logger
    logger = svdc.configureLogging('SVD_Approach3.log', args['loglevel'], args['console'])
    if(args['pickle']):
        svdc.convert2Pickle(args['dataset'][0])
    if(args['train']):
        trainModel()

args = None
logger = None

if __name__=="__main__": 
    main() 