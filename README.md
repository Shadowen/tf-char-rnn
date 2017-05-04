# Instructions
Run any `*Experiment.py` file with `python3` to train and evaluate a model. Models will automatically save to and load from `models/*`.

## Samples


### [VanillaRNNExperiment](VanillaRNNExperiment.py)
1-layer RNN with 128 cells. 25 step BPTT unrolling.

    PEREY:
    Seld tell sorst her that uptentare,
    With your
    Beast to threal undrenciust is thou mank' meany Sufer she, reased, friend
    First not so now sach dan ungaluble
    Well, therey all constie,
    That I mwer wakerth Duke am is thangucth'd atnay vecantes; your gare that I'll-riber,
    You ha't not sighty have fromy I vouch a knocks known of Come: good wind whom that I lein of the bow dobe to then, kishand their comestry a shor'd hims lour,
    Af, these not scapry tend that!
    
    SCYMILCUT:
    Youthou'ly not, all, How dud turs: honour'd; by gone,
    Vight it,
    If gold doggurs,
    For I still but what! what, my noul dayanch is upan him sear to than harious, which you what was he houlds good be.
    
    SILIALIAN HARCUS:
    So I know ot diggaing withbus to thy wild thou all, a drain's a luter,
    Ang the now foit fresh, better fallow

### [MultiLSTMExperiment](MultiLSTMExperiment.py)
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
- Timing analysis
- Calculate perplexity
- Clean up get_unique_deterministic by saving the mappings inside RNNEstimator
- Sample with temperature using tf.multinomial
- Embedding with tensorflow with tf.embedding_lookup