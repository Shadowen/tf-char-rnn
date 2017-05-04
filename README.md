# Instructions
Run any `*Experiment.py` file with `python3` to train and evaluate a model. Models will automatically save to and load from `models/*`.

## Samples
### [VanillaRNNExperiment](VanillaRNNExperiment.py)
RNN with 128 cells. 25 step BPTT unrolling.
    
    DUCHESS OF Gade anstle
    Anow to services on true kill's.
    Nor thou cospity any sing.
    
    HERMIONE:
    Noye, To cupity:
    You dear lemiout like not him,

### [MultiRNNExperiment](MutliRNNExperiment.py)
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

    
### [EpochBasedMultiLSTMExperiment](EpochBasedMultiLSTMExperiment.py)
3-layer LSTM with 512 cells each. 0.5 dropout_keep_prob on RNN outputs during training. 100 step BPTT unrolling.

    Good God: and thou'lt not put thee in thy
    requisation. Say that ave your father shall joy--
    
    LORD POLONIUS:
    You were man; ay, you do, it is a noble potent to the
    creature. Wherein all bid 'To this gipsy's shallows to
    the head of door how? doth you forth but a stage?
    
    Boy:
    Or not an lord, if thou didst deliver a thousand deeds.
    
    DUKE ORSINO:
    I will help you, gentlemen.
    
    BOYET:
    Princes, it is no more of my life! I think I am not
    warm, here is Normandy. The toing ere thy tongue
    be, should forthwith flex or forfeit by infancy, but
    cross-breaking house. I'll no more to you.
    
    TITANIA:
    Receive him; you, my good lord, come to leave,
    To see them master of her tongue; for he is so much
    much very enough.
    
    MALVOLIO:
    So thou shalt create; thou'rt not Banquo, and mistress
    of Nature; and your stales against thy death. Follow
    it: stay, man! goddess!
    
# TODO
- RNN memory visualization
- Summaries
- Timing analysis
- Clean up get_unique_deterministic by saving the mappings inside RNNEstimator
- Sample with temperature using tf.multinomial
- Embedding with tensorflow with tf.embedding_lookup