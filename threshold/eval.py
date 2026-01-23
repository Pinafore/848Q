# Jordan Boyd-Graber
# 2023
#
# Run an evaluation on a QA system and print results
import random
import string
import logging

from tqdm import tqdm

from parameters import load_guesser, load_questions, load_buzzer, \
    add_buzzer_params, add_guesser_params, add_general_params,\
    add_question_params, setup_logging

kLABELS = {"best": "Guess was correct, Buzz was correct",
           "timid": "Guess was correct, Buzz was not",
           "hit": "Guesser ranked right page first",
           "close": "Guesser had correct answer in top n list",
           "miss": "Guesser did not have correct answer in top n list",
           "aggressive": "Guess was wrong, Buzz was wrong",
           "waiting": "Guess was wrong, Buzz was correct"}

def expected_win_probability(percentage):
    """
    https://arxiv.org/pdf/1904.04792
    """
    t = percentage
    return 1 + 0.0775*t - 1.278*t**2 + 0.588*t**3

def normalize_answer(answer):
    """
    Remove superflous components to create a normalized form of an answer that
    can be more easily compared.
    """
    from unidecode import unidecode
    
    if answer is None:
        return ''
    reduced = unidecode(answer)
    reduced = reduced.replace("_", " ")
    if "(" in reduced:
        reduced = reduced.split("(")[0]
    reduced = "".join(x for x in reduced.lower() if x not in string.punctuation)
    reduced = reduced.strip()

    for bad_start in ["the ", "a ", "an "]:
        if reduced.startswith(bad_start):
            reduced = reduced[len(bad_start):]
    return reduced.strip()
 
def rough_compare(guess, page, trace=None):
    """
    See if a guess is correct.  Not perfect, but better than direct string
    comparison.  Allows for slight variation.
    """
    # TODO: Also add the original answer line
    if page is None:
        return False
    
    guess = normalize_answer(guess)
    page = normalize_answer(page)

    if guess == '':
        return False
    
    if guess == page:
        return True
    elif page.find(guess) >= 0 and (len(page) - len(guess)) / len(page) > 0.5:
        return True
    else:
        return False
    
def eval_retrieval(guesser, questions, n_guesses=25, cutoff=-1):
    """
    Evaluate the guesser's retrieval
    """
    from collections import Counter, defaultdict
    outcomes = Counter()
    examples = defaultdict(list)
    confidence = []

    question_text = []
    for question in tqdm(questions):
        text = question["text"]
        if cutoff == 0:
            text = text[:int(random.random() * len(text))]
        elif cutoff > 0:
            text = text[:cutoff]
        question_text.append(text)

    all_guesses = guesser.batch_guess(question_text, n_guesses)
    assert len(all_guesses) == len(question_text)
    for question, guesses, text in zip(questions, all_guesses, question_text):
        if len(guesses) > n_guesses:
            logging.warning("Warning: guesser is not obeying n_guesses argument")
            guesses = guesses[:n_guesses]
            
        top_guess = guesses[0]["guess"]
        answer = question["page"]
        assert guesses[0]["confidence"] >= 0.0, "Negative confidence %0.2f" % guesses[0]["confidence"]

        example = {"text": text, "guess": top_guess, "answer": answer, "id": question["qanta_id"]}
        print(example)

        if any(rough_compare(x["guess"], answer) for x in guesses):
            outcomes["close"] += 1
            if rough_compare(top_guess, answer):
                outcomes["hit"] += 1
                examples["hit"].append(example)
                confidence.append(guesses[0]["confidence"])
            else:
                examples["close"].append(example)
                confidence.append(-guesses[0]["confidence"])                
        else:
            outcomes["miss"] += 1
            examples["miss"].append(example)

    return outcomes, examples, confidence

def pretty_feature_print(features, first_features=["guess", "answer", "id"]):
    """
    Nicely print a buzzer example's features
    """
    
    import textwrap
    wrapper = textwrap.TextWrapper()

    lines = []

    for ii in first_features:
        lines.append("%20s: %s" % (ii, features[ii]))
    for ii in [x for x in features if x not in first_features]:
        if isinstance(features[ii], str):
            if len(features[ii]) > 70:
                long_line = "%20s: %s" % (ii, "\n                      ".join(wrapper.wrap(features[ii])))
                lines.append(long_line)
            else:
                lines.append("%20s: %s" % (ii, features[ii]))
        elif isinstance(features[ii], float):
            lines.append("%20s: %0.4f" % (ii, features[ii]))
        else:
            lines.append("%20s: %s" % (ii, str(features[ii])))
    lines.append("--------------------")
    return "\n".join(lines)

def eval_buzzer(buzzer, questions, history_length, history_depth):
    """
    Compute buzzer outcomes on a dataset
    """
    
    from collections import Counter, defaultdict

    # the buzzer will destroy the full text of the questions to protect from
    # cheating (as it should), so we compute the question length now
    question_max_length = defaultdict(int)
    run_lengths = []
    for question in questions:
        qid = question["qanta_id"]
        run_length = len(question["text"])
        question_max_length[qid] = max(question_max_length.get(qid, 0), run_length)
        run_lengths.append(run_length)
    
    buzzer.load()
    buzzer.add_data(questions)
    buzzer.build_features(history_length=history_length, history_depth=history_depth)
    
    predict, feature_matrix, feature_dict, correct, metadata = buzzer.predict(questions)
    
    outcomes = Counter()
    examples = defaultdict(list)
    expected_wins = []
    neg_on_question = defaultdict(bool)
    question_seen = {}
    question_length = defaultdict(int)
    
    if len(feature_dict) < len(predict):
        logging.warning("Empty feature dict (this is normal and expected for buzzers like LoraBERT)")
        feature_dict = [{"buzz": x} for x in predict]
    
    for buzz, guess_correct, features, meta, run_length in zip(predict, correct, feature_dict, metadata, run_lengths):
        # TODO: Fix the feature-based pipelines and the BERT pipelines to use consistent identifiers
        try:
            qid = meta["qanta_id"]
        except KeyError:
            qid = meta["id"]
        
        # Add back in metadata now that we have prevented cheating in feature creation        
        for ii in meta:
            features[ii] = meta[ii]

        assert qid in question_max_length
        percentage = run_length / question_max_length[qid]

        question_length[qid] = max(question_length[qid], len(meta["text"]))
        
        if guess_correct:
            if buzz:
                if not neg_on_question[qid]:
                    expected_wins.append(expected_win_probability(percentage))
                else:
                    expected_wins.append(-3)
                outcomes["best"] += 1
                examples["best"].append(features)
                if not qid in question_seen:
                    question_seen[qid] = len(meta["text"])
            else:
                # Because all negative expected wins will be thrown out, let's track state with the following scheme:
                # -1: Waiting without neg
                # -1.5: Waiting after neg
                # -2: Negging for the first time
                # -3: You would have buzzed correctly, but it's after neg
                if neg_on_question[qid]:
                    expected_wins.append(-1.5)
                else:
                    expected_wins.append(-1)
                outcomes["timid"] += 1
                examples["timid"].append(features)
        else:
            if buzz:
                expected_wins.append(-2)
                neg_on_question[qid] = True
                outcomes["aggressive"] += 1
                examples["aggressive"].append(features)
                if not qid in question_seen:
                    question_seen[qid] = -len(meta["text"])
            else:
                if neg_on_question[qid]:
                    expected_wins.append(-1.5)
                else:
                    expected_wins.append(-1)
                outcomes["waiting"] += 1
                examples["waiting"].append(features)
    
    unseen_characters = 0.0

    number_questions = 0
    for question in question_length:
        number_questions += 1
        length = question_length[question]
        if question in question_seen:
            if question_seen[question] > 0:
                # The guess was correct
                unseen_characters += 1.0 - question_seen[question] / length
            else:
                unseen_characters -= 1.0 + question_seen[question] / length

    assert len(expected_wins) > 0, "Did not get any questions"

    return outcomes, examples, unseen_characters / number_questions, sum(x for x in expected_wins if x >= 0) / len(question_max_length)
                
def calibration(confidences, epsilon=0.001):
    max_conf = max(abs(x) for x in confidences)
    min_conf = min(abs(x) for x in confidences)

    calibration_error = [(1 - ((x - min_conf) / (max_conf - min_conf + epsilon)))**2 if x > 0 else ((x - min_conf) / (max_conf - min_conf + epsilon))**2 for x in confidences]

    return sum(calibration_error) / len(calibration_error)

if __name__ == "__main__":
    # Load model and evaluate it
    import argparse
    
    parser = argparse.ArgumentParser()
    add_general_params(parser)
    guesser_params = add_guesser_params(parser)
    question_params = add_question_params(parser)
    buzzer_params = add_buzzer_params(parser)

    parser.add_argument('--evaluate', default="buzzer", type=str)
    parser.add_argument('--cutoff', default=-1, type=int)    
    
    flags = parser.parse_args()
    setup_logging(flags)

    questions = load_questions(flags)
    guesser = load_guesser(flags, guesser_params, load=flags.load)    
    if flags.evaluate == "buzzer":
        buzzer = load_buzzer(flags, buzzer_params, load=True)
        outcomes, examples, unseen, ew = eval_buzzer(buzzer, questions,
                                             history_length=flags.buzzer_history_length,
                                             history_depth=flags.buzzer_history_depth)
    elif flags.evaluate == "guesser":
        if flags.cutoff >= 0:
            outcomes, examples, confidences = eval_retrieval(guesser, questions, flags.num_guesses, flags.cutoff)
        else:
            outcomes, examples, confidences = eval_retrieval(guesser, questions, flags.num_guesses)
    else:
        assert False, "Gotta evaluate something"
        
    total = sum(outcomes[x] for x in outcomes if x != "hit")
    for ii in outcomes:
        print("%s %0.2f\n===================\n" % (ii, outcomes[ii] / total))
        if len(examples[ii]) > 10:
            population = list(random.sample(examples[ii], 10))
        else:
            population = examples[ii]
        for jj in population:
            print(pretty_feature_print(jj))
        print("=================")
        
    if flags.evaluate == "buzzer":
        if len(buzzer._classifier.coef_[0]) > 0:
            for weight, feature in zip(buzzer._classifier.coef_[0], buzzer._featurizer.feature_names_):
                print("%40s: %0.4f" % (feature.strip(), weight))
        
        print("Questions Right: %i (out of %i) Accuracy: %0.2f  Buzz ratio: %0.2f Buzz position: %f Expected Wins: %f" %
              (outcomes["best"], # Right
               total,            # Total
               (outcomes["best"] + outcomes["waiting"]) / total, # Accuracy
               (outcomes["best"] - outcomes["aggressive"] * 0.5) / total, # Ratio
               unseen, # position
               ew))
    elif flags.evaluate == "guesser":
        print("Precision @1: %0.4f Recall: %0.4f Calibration Error: %0.4f" %
              (outcomes["hit"]/total, outcomes["close"]/total, calibration(confidences)))
