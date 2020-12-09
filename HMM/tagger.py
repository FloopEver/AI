import numpy as np
from hmm import HMM


def model_training(train_data, tags):
    """
    Train an HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class 
            defined in data_process.py (read the file to see what attributes this class has)
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - model: an object of HMM class initialized with parameters (pi, A, B, obs_dict, state_dict) calculated 
            based on the training dataset
    """

    # unique_words.keys() contains all unique words
    unique_words = get_unique_words(train_data)

    word2idx = {}
    tag2idx = dict()
    S = len(tags)
    ###################################################
    # TODO: build two dictionaries
    #   - from a word to its index 
    #   - from a tag to its index 
    # The order you index the word/tag does not matter, 
    # as long as the indices are 0, 1, 2, ...
    ###################################################
    i = 0
    for word in unique_words:
        word2idx[word] = i
        i += 1
    j = 0
    for tag in tags:
        tag2idx[tag] = j
        j += 1

    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))
    ###################################################
    # TODO: estimate pi, A, B from the training data.
    #   When estimating the entries of A and B, if  
    #   "divided by zero" is encountered, set the entry 
    #   to be zero.
    ###################################################
    states_transition = []
    observations = []
    for line in train_data:
        tag = line.tags[0]
        states_transition += [(tag1, tag2) for tag1, tag2 in zip(line.tags, line.tags[1:])]
        observations += [(tagi, wordj) for tagi, wordj in zip(line.tags, line.words)]
        pi[tag2idx[tag]] += 1
    pi /= np.sum(pi)

    for (tag1, tag2) in states_transition:
        A[tag2idx[tag1], tag2idx[tag2]] += 1
    A_sum = np.sum(A, axis=1)
    for i in range(S):
        if A_sum[i] != 0:
            A[i] /= A_sum[i]

    for (tagi, wordj) in observations:
        B[tag2idx[tagi], word2idx[wordj]] += 1
    B_sum = np.sum(B, axis=1)
    for i in range(S):
        if B_sum[i] != 0:
            B[i] /= B_sum[i]


    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, word2idx, tag2idx)
    return model


def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
    - model: an object of the HMM class
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ######################################################################
    # TODO: for each sentence, find its tagging using Viterbi algorithm.
    #    Note that when encountering an unseen word not in the HMM model,
    #    you need to add this word to model.obs_dict, and expand model.B
    #    accordingly with value 1e-6.
    ######################################################################

    for line in test_data:
        obq = line.words
        for w in obq:
            if w not in model.obs_dict:
                model.obs_dict[w] = max(model.obs_dict.values()) + 1
                new_col = np.zeros(len(tags)) + 1e-6
                model.B = np.c_[model.B, new_col]
                B_sum = np.sum(model.B, axis=1)
                for i in range(len(tags)):
                    if B_sum[i] != 0:
                        model.B[i] /= B_sum[i]
        tagging.append(model.viterbi(obq))

    return tagging


# DO NOT MODIFY BELOW
def get_unique_words(data):
    unique_words = {}

    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words
