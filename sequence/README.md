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
 "Guesser" class that respects the underlying API.

 2. Make an improvement to the buzzer that uses sequence modeling in
 some way.  This must be a subclass of the generic "Buzzer" class that
 respects the underlying API.

Don't try to be too ambitious.  Focus on an approach that you can get
done quickly and then slowly build on it to improve it into something
better.  Remember that you'll need to have this run on Gradescope, so
things that are too computationally expensive won't work.

What You're Not Allowed to Do
=============================

  1.  Do anything that is so complicated that it cannot run on
      GradeScope (this rules out most Muppet Models).
  2.  Break the underlying API (otherwise, you can't be on the leaderboad)
  3.  Use QANTA test data in training your code.

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