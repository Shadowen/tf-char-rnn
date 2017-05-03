# Instructions
Run any `*Experiment.py` file with `python3` to train and evaluate a model. Models will automatically save to and load from `models/*`.

## Samples
### [VanillaRNNExperiment](VanillaRNNExperiment.py)
RNN with 128 cells. 25 step BPTT unrolling.

    SICINIUS:
    So, see: thou brook! your book! God, Marcius King be are the isseeped,
    Steantly, good for consonerio, as to great mock till I have me'er 'ccoune!
    Home pation of weet-slains. She the chaugesty, brown than you, pearies and bank ladan;
    And it, and marrer yet would of think the seoveder have place with manysled
    Here is hence; or they noths
    Of the such me, and against is illos; with upoin dulply, for him; this in his enaubfletions,
    I will no may in the king ack,
    When dumbll-dis he should?
    
    Muss'd, let expect.
    
    DUCHESS OF Gade anstle
    Anow to services on true kill's.
    Nor thou cospity any sing.
    
    HERMIONE:
    Noye, To cupity:
    You dear lemiout like not him,
    I do be was dobocks at sound, anghoo the shalt,
    My son; of slewily antiul to the fleed thee since
    If I how turn, and put an earfretition

### [MutliRNNExperiment](MutliRNNExperiment.py)
2-layer RNN with 128 cells each. 25 step BPTT unrolling.

    CYSUS:
    Heaven gives you in love; welcome again; there in thy lord, and poor water?
    Farewell.
    
    ALBANY:
    The stone entugin the Dayed attretour condowires. Madam,
    Heaven forfing King Lord:
    Set, Deing that is Lypimness halters,
    In guilt, to me if I am a
    vour moor.
    
    FLAVIUS:
    Revenbs Clite
    It well not did too love;
    Whileful please for your
    perdomirazed by Pagars.

### [MultiLSTMExperiment](MultiLSTMExperiment.py)
3-layer LSTM with 512 cells each. 0.5 dropout_keep_prob on RNN outputs during training. 50 step BPTT unrolling.

    PROSPERO:
    from this young beds; being better ladies, here that all the
    hall you well see Mountremen?
    
    FALSTAFF:
    So a scorming of good tent hath slung him when I am I'll be committed to be is not in the gacked before!
    If he report on me my joyful virtue; hath
    so state honoured, but therefore we swift fair lands thou dost meet.
    
    FALSTAFF:
    Pray God, my lord; it was a villain, good madam; for be a thankful looks out of the way kind on priest;
    Never sounded will, last relieved.
    
    Most right; servantly.
    
    MOWBRAY:
    Very blind, my liege, I must show it for hin good,
    And are past promited, and tell me there.
    
    ANTIPHOLUS OF SYRACUSE:
    I will, my lord: I for a well-dignity, that Had all the act,
    Hast thou you trumpet in Thursday people! the queen?
    Farewell, lady.'

# TODO
- RNN memory visualization
- Summaries
- Clean up get_unique_deterministic by saving the mappings inside RNNEstimator
- Sample with temperature using tf.multinomial
- Embedding with tensorflow with tf.embedding_lookup