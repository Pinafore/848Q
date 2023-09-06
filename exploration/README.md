Representations and Sequences
-----------------------------

While the previous homeworks held your hand considerably, this
homework will not.  Some of you will enjoy the freedom, others of you
will complain about not being told exactly what to do.  To the latter
group, we will give you some ideas of what you can do.  To the former
group, do not feel constrained by these suggestions.  If you want to
do something different, run it by us and we'll likely say okay.

What You Have to Do
===================

There are two requirements on the substance of what you must turn in:

 1. Make an improvement to the guesser that uses representation
 learning in some way.  This must be a subclass of the generic
 "Guesser" class that respects the underlying API.  This could be a
 Muppet model or something simpler.

 2. Make an improvement to the buzzer that uses sequence modeling in
 some way.  This must be a subclass of the generic "Buzzer" class that
 respects the underlying API.

Don't try to be too ambitious.  Focus on an approach that you can get
done quickly and then slowly build on it to improve it into something
better.  Remember that you'll need to have this run on Gradescope, so
things that are too computationally expensive won't work.

What You're Not Allowed to Do
=============================

  1.  Break the underlying API (otherwise, you can't be on the leaderboad)
  2.  Use QANTA test data in training your code.

Getting things to Work
===============

While you will have access to Nexus resources for *training*, at
*test* time you're still going to need to run the code on CPU-only
Gradescope.

So, you may want to consider smaller models, avoiding large generative
models, or distilling your models to ensure they work in more
constrained environments.

We're not just doing this because we're lazy: an important part of
having a project work well is having simple enough code that will run
quickly and that you can get answers from.  So you need to think about
the simplest approach that could work, test the hypothesis, and get
results.  This will hopefully serve you well for your project.

Suggestions
===========

For representation learning:

  1.  Use pre-trained embeddings as an additional feature to rerank answers in the Guesser  
  2.  Implement a simple averaging or convolutional feature (link to
  undergraduate
  homework)[https://github.com/Pinafore/nlp-hw/tree/master/dan]

For sequence learning:

  1.  Look at the sequence of guesses / confidences to do a better job
  of buzzing
  2.  Use a named entity tagger to identify entities in
  questions and use the corresponding Wikipedia page as additinal
  evidence (e.g., if the question mentions the Grand Canal, you can
  see that page mentions the Sui Dynasty, and you can add that as an
  additional guess).
  3.  Parse the questions and use that information to improve the
  guesser or buzzer.
  4.  Change the buzzer to user the guess history and feed it into a model like [crf-suite](https://pypi.org/project/python-crfsuite/).

For Muppet models:

 1. Fine tuning BERT to create a better buzzer

 2. Using a Muppet Model as a guesser: create a softmax layer over
   closed set (e.g., Wikipedia page titles) with a single softmax
   layer.  This is probably the easiest way to do it.

 3. Using a Muppet Model as a retriever: fine-tune DPR or some other
   deep retrieval model to specifically retrieve corresponding
   evidence passage and then either answer with the page title or ...

 4. Using a Muppet Model as a machine reading model.  Given an evidence
   passage (either generated from a deep retriever or from, say, the
   tf-idf homework), use that to generate a guess string from the
   evidence.

 5. Using a Muppet Model as a generative LM guesser.  You could use APIs for
   this, which should be pretty easy (we have [cached GPT3
   guesses](https://github.com/Pinafore/848Q/blob/main/models/gpt_cache.tar.gz)
   and [a guesser that uses those
   guesses](https://github.com/Pinafore/848Q/blob/main/muppet/gpr_guesser.py)
   if you want to train a downstream buzzer on that).  The *hard mode* version
   of this would be to use DPO to make it work better specifically on this
   dataset.

You can do something else too!  The ideal would be to find something
that would connect to the kinds of ideas you'd do in your final
project.

Groups
======

While it's allowed to do this on your own, it's highly recommended you
work on this in groups, since there are clear discrete components
where it's easier to "share the burden".

What you Have to Upload
=======================

Upload code that runs on the autograder (e.g., if you train a model,
include a pickle file / model file so that it can be run directly).
Again, Gradescope has a maximum upload size as well, so your model
should be less than 200MB.

Also upload a writeup explaining what you did (with citations as
appropriate).  The writeup should be comprehensive enough that the TA
should be able to understand your code (this is a good reason to keep
the things you try simple).

Grading
=======

30 points are available for satisfying the spirit of the assignment,
explaining what you did, and documenting the code well enough for the
TA to understand what's going on.

10 additional bonus points are available for doing particularly well on the
leaderboard or for being particularly clever.


Leaderboards
=========

Because this is more computationally expensive, we are going to have
three leaderboards.  One where you just upload your predictions, one
where we run your code on a small number of predictions, and one where
we run your code on a larger test set.

Example
======

Here I'm going to walk through creating a new feature that looks at
the `guess_history`.

First, let's add the feature (to `features.py`) and try it out on a
tiny example to make sure that it works:

    python3 buzzer.py --limit=10 --questions=data/qanta.buzztrain.json.gz --num_guesses=10 --buzzer_history_length=5 --buzzer_history_depth=3 --features Length PreviousGuess
