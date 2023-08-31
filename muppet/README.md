
Muppet Models
-------------


Getting things to Work
======================

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

Ideas
=====

Like the last assignment, this is open-ended.  You can do anything
that uses large language models.  Okay, the first question is "what's
a large langauge model?".  For the purposes of this assignment, a
large language model is:

  * A language model (with any structure / model) trained on more than
    100M tokens of text.  This includes Knesser-Ney, Pitman-Yor, etc.
  
  * A language model that uses word representations and transformers
    (i.e., a more traditional Muppet Model).

You *do not* need to use a generative language model.  Indeed, I'd
probably discourage you from doing that, as it's probably not going to
be something you can get working in the time frame required.

Here are some things you could do and would be worth full credit:

 * Fine tuning BERT to create a better buzzer

 * Using a Muppet Model as a guesser: create a softmax layer over
   closed set (e.g., Wikipedia page titles) with a single softmax
   layer.  This is probably the easiest way to do it.

 * Using a Muppet Model as a retriever: fine-tune DPR or some other
   deep retrieval model to specifically retrieve corresponding
   evidence passage and then either answer with the page title or ...

 * Using a Muppet Model as a machine reading model.  Given an evidence
   passage (either generated from a deep retriever or from, say, the
   tf-idf homework), use that to generate a guess string from the
   evidence.

 * Using a Muppet Model as a generative LM guesser.  You could use APIs for
   this, which should be pretty easy (we have [cached GPT3
   guesses](https://github.com/Pinafore/848Q/blob/main/models/gpt_cache.tar.gz)
   and [a guesser that uses those
   guesses](https://github.com/Pinafore/848Q/blob/main/muppet/gpr_guesser.py)
   if you want to train a downstream buzzer on that).  The *hard mode* version
   of this would be to use DPO to make it work better specifically on this
   dataset.

Leaderboards
============

Because this is more computationally expensive, we are going to have
three leaderboards.  One where you just upload your predictions, one
where we run your code on a small number of predictions, and one where
we run your code on a larger test set.

Grading
=======

30 points are available for satisfying the spirit of the assignment,
explaining what you did, and documenting the code well enough for the
TA to understand what's going on.

Obviously, it's best if you can do all the leaderboards!  But you are
only required to do it on the "upload predictions" leaderboard.
