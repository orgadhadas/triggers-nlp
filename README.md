# triggers-nlp: Seminar project code
This is the code for the project about weakness of trigger attack that proposed in Universal Adversarial Triggers for Attacking and Analyzing NLP by Wallace et al.

Installation instructions:

Create new environment and use the requirements.txt file to install all packages.
You can use the following line:

    conda create -n seminar_project python=3.7
      
    conda activate seminar_project
      
    pip install -r requirements.txt

The structure of the git folder:
    
    triggers_used: File that contains all the triggers we used when we tested our model.
    data folder: Contains three clean trining files. Each file contains sentences from the training set snli dataset: 
    Each file contains sentences from diffrent class (contardiction, entailment and neutral). We make sure that there 
    are not sentences that repeat themself. The file names are: train_data_label_contradiction_uniq.txt, 
    train_data_label_entailment_uniq.txt and train_data_label_neutral_uniq.txt. Another file that is in this folder is 
    train_data_label_entailment.txt.cat that contains the sentences from the train_data_label_entailment_uniq.txt but the
    word 'cat' inserted at the beginning of each sentence.
    This folder also consist trigger_data folder which describes next. 
    data/triggerd_data: Folder that contains 25 files. The name of each file starts with "dev_data_label_entailment_uniq.txt.".
    The file name ends with trigger words (seperated by '_') that inserted at the beginning of each sentences in the file.
    The sentences in the file are sentences from the contradiction class in the dev snli dataset. The trigger that we 
    used tries to change the sentences from contradiction to entailment.
    src folder: Contains scripts used to run the experiments described in the report and the output of some of these
    scripts. The files are described in the next section


We conduct three experiments as described in the paper. In each experiment we use different model:

    1) bigram model for detection
    
    2) GPT model 
    
    3) RoBERTa
    
**1) Running bigram model:** In order to run this experiment you need to run bi_gram.py script with the following flags:
    
    --dir: A path to a folder where there are files that at the beginning of each sentence have a trigger. In our 
    experements we gave ./data/triggered_data as input.
    --clean: File with out the trigger. In our experiment ./data/dev_data_label_entailment_uniq.txt was given 
    The script uses the file train_data_label_entailment_uniq.txt that is in the data folder as input to bigram model.
    The bigram is tested on the triggered files that in dir input and the clean input.

**2) Running GPT2 model:** In order to run this experiment you first to run get_sentence_statistics.py. The flags to run
    the scripts are:
      
    --input_path: A path to a file that contains sentences from which statistics are collected. In our experiments we 
    used the training set with entailment label ./data/train_data_label_entailment_uniq.txt
    --ouput_path: A path to a file where the statistics are saved in json format. We saved the statistics in the file:
    train_entailment_stats.json.  
    
   After we get the statistic file we use it to run detection_GPT2.py script with the following flags:

    --dir: A path to a folder where there are files that at the beginning of each sentence have a trigger. In our 
    experements we gave ./data/triggered_data as input.
    --clean: File without the trigger. In our experiment ./data/dev_data_label_entailment_uniq.txt was given 
    The script uses the file train_data_label_entailment_uniq.txt that is in the data folder as input to bigram model.
    The bigram is tested on the triggered files that in dir input and the clean input.
    --stats: A json file that contains statistic about the loss of GPT model. This file is the output of 
    get_sentence_statistics.py. As we said we saved the statistics in train_entailment_stats.json.
  
 **3) Running RoBERTa model:** In order to run this experiment you first to run get_sentence_statistics_roberta.py.
  The flag that has to be provided is:
   
    --file which determines the file with to get statistics on. In this experement we used train_data_label_entailment.txt.cat
    which can be find in the data folder 
   The output file in this experiment is: roberta_stats.pickle
  
  After we get the statistic file we use it to run roberta.py script with the following flags:
  
    --dir: A path to a folder where there are files that at the beginning of each sentence have a trigger. In our 
    experements we gave ./data/triggered_data as input.
    --clean: File with out the trigger. In our experiment ./data/dev_data_label_entailment_uniq.txt was given 
    The script uses the file train_data_label_entailment_uniq.txt that is in the data folder as input to bigram model.
    The bigram is tested on the triggered files that in dir input and the clean input.
    --stats: A pickle file that contains statistic about the loss of RoBERTa model. This file is the output of 
    get_sentence_statistics_roberta. As we said we saved the statistics in roberta_stats.pickle.
  
   
 