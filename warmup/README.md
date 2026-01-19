
Threshold Buzzer
=======================

To get us warmed up, we're going to use an existing code repository that tries to answer questions.

This homework is meant to be very easy.  You should not need to spend a lot of
time on coding.  The hardest part of this homework is understanding the existing code, which we'll use for future homeworks.

However, like many of our homework assignments, there will be opportunity for
extra credit.

What you have to do
===================

The TfIdf takes a question and returns a guess of who the president
was.  You need to extract what time the question is asking about and return
a promising answer.  You should use sklearn's tf-idf retrieval; this should be straightforward.  The code for processing the data is provided, you should understand it but not modify it.

The problem is when to trust that answer.  The `ThresholdBuzzer` class decides when to trust that answer.  Inspect the output of your guesser class and try to find a good value for the two parameters that control when to buzz in: how much of the question needs to be revealed before it trusts the answer and how high the score of the buzzer needs to be before it answers.

This should be very simple, no more than five lines of code.  If you're
writing far more than that, you're likely not taking advantage of built-in
Python libraries that you should be using.

How do I know if my code is working?
====================================

Run `eval.py` to see if you're answering too early or too late.

How to turn it in
=================

Modify the two files `threshold_buzzer.py` and `tfidf_guesser.py` and upload them HW0 on Gradescope.

Extra Credit
============

Rather than setting the values manually, complete the `train` function in `threshold_buzzer` to learn the value from data.

Frequently Asked Questions
==========================


Points Possible
===============

You get full credit (seven points) for matching the baseline accuracy (85%) and can get up to
three points for improving significantly beyond that.
