import logging

from collections import Counter, defaultdict

from parameters import Parameters
from guesser import Guesser
from gpr_guesser import kINSTRUCTIONS

import dspy

class OllamaGuesser(Guesser):
    # TODO(jbg): Make it so that this can cache like GPR guesser, but also include the name of the model
    def __init__(self):        
        self._model = None
    
    def set_ollama_model(self, model, instructions=kINSTRUCTIONS):
        self._model = model

        self._instructions = instructions

    def query(self, question):
        from ollama import chat
        from ollama import ChatResponse
        
        assert self._model is not None, "You need to set the model first"

        response: ChatResponse = chat(model='gemma3:4b', messages=[
            {
                'role': 'user',
                'content': self._instructions  + "\n\nQuestion %s" % question,
            },
        ])
        
        return response

def validate_answer(example: dspy.Example, pred, trace=None):
    from eval import rough_compare

    correct = rough_compare(example.answer, pred.best_guess)

    retrieval_bonus = 0.0

    # This part should provide a signal to the query generation
    for guess in pred.other_guesses:
        if rough_compare(guess, example.answer):
            retrieval_bonus += 1
        
    # Normalize by number of guessers from tf-idf
    if len(pred.other_guesses) > 0:
        retrieval_bonus /= len(pred.other_guesses)

    logging.info("Retrieval bonus = %0.2f from  ans=%s clues=%s" % (retrieval_bonus, example.answer, ", ".join(x for x in pred.other_guesses)))
        
    # Bonus for correct confidence 
    if correct:
        return 1 + pred.confidence ** 2 + retrieval_bonus
    else:
        return retrieval_bonus - pred.confidence ** 2

   
class CalibratedRAG(dspy.Module):
    def __init__(self):

        class QueryGenerator(dspy.Signature):
            """Given a question, create a list of queries that could find information that would support or refute candidate answers"""

            question: str = dspy.InputField()
            queries: list[str] = dspy.OutputField(desc="Often combines a possible guess with information in the question")

        class PredictionGenerator(dspy.Signature):
            """Given a series of queries to an IR system and their results, generate a list of possible guesses that could answer the question and your confidence they are correct.  Include guesses eliminated by contradictory information by giving them a confidence of 0.0"""
            question: str = dspy.InputField()
            queries: list[str] = dspy.InputField()
            contexts: list[str] = dspy.InputField(desc="Wikipedia information and past questions that can contain information about possible guesses")
            intermediate_guesses: list[str] = dspy.OutputField(desc="A diverse set of guesses to the answer to the question inspired by the retrieved information, removing duplicate guesses or alternate names")
            intermediate_confidences: list[float] = dspy.OutputField(desc="A probability for each guess and whether it is the correct answer taking into account the length of the question and the strength of the evidence.  Should sum to 1.0 or less")
            intermediate_reasonings: list[str] = dspy.OutputField(desc="An evidence-based rationale for why this guess is correct (or not), focusing on relationships found from the retrieved contexts")

        class FullResult(dspy.Signature):
            """Given the set of guesses, create a final weighted guess of the probability of the best guess being correct"""
            question: str = dspy.InputField(desc="The question we're trying to answer")
            final_guesses: list[str] = dspy.InputField(desc="The best guess of the model given the relevant information")
            final_confidences: list[float] = dspy.InputField(desc="A probability for each guess and whether it is the correct answer.  Should sum to 1.0 or less")
            final_reasonings: list[str] = dspy.InputField(desc="The best guess of the model given the relevant information")
            other_guesses: list[str] = dspy.OutputField(desc="Other guesses considered (copied directly from input)")
            best_guess: str = dspy.OutputField(desc="Single best guess of correct answer of the model")
            confidence: float = dspy.OutputField(desc="Probability that the best guess is correct")
            reasoning: str = dspy.OutputField(desc="Short rationale for confidence and guess without mentioning guess")

        self.topk = None
        self.retrievers = {}
        self.query_generator = dspy.Predict(QueryGenerator)
        self.guess_generator = dspy.ChainOfThought(PredictionGenerator)
        self.confidence_generator = dspy.Predict(FullResult)

    def forward(self, question, **kwargs):
        
        queries = self.query_generator(question=question).queries
        contexts = []

        for query in queries:           
            context = ""
            for retriever in self.retrievers:
                results = self.retrievers[retriever](query, self._topk)
                for result in results:
                    context += "%s: %s: %s\n" % (result["guess"], retriever, result["question"])
            contexts.append(queries)
            
        guess_result = self.guess_generator(queries=queries, contexts=contexts, question=question)

        # Make the guesses unique if the model didn't do it already
        guess_probability = defaultdict(float)
        reasons = defaultdict(str)
        normalizer = sum(guess_result.intermediate_confidences)
        assert min(guess_result.intermediate_confidences) >= 0.0, "Negative confidence"

        for guess, confidence, reason in zip(guess_result.intermediate_guesses, 
                                             guess_result.intermediate_confidences,
                                             guess_result.intermediate_reasonings):
            guess_probability[guess] += confidence / normalizer
            reasons[guess] += "%s\n" % reason
        sorted_guesses = sorted(guess_probability.keys(), key=lambda x: guess_probability[x])

        fg = sorted_guesses
        fr = [reasons[x] for x in sorted_guesses]
        fc = [guess_probability[x] for x in sorted_guesses]

        print(fg, fr, fc)

        return self.confidence_generator(question=question,
                                         final_guesses=fg, 
                                         final_confidences=fc,
                                         final_reasonings=fr)

    def init_retriever(self, name:str, model_filename:str, topk:int=1):
        """
        Add a new retriever to the system.

        Parameters:
        - name: The key of the retriever
        - filename: Where to load the retriever from (filename)
        - topk: How many retrieved results to get from the retriever per query
        """
        
        logging.info("Loading retriever %s as %s (top k=%i)" % (model_filename, name, topk))  
        from tfidf_guesser import TfidfGuesser 
        retriever = TfidfGuesser(model_filename)
        self.retrievers[name] = retriever 
        self.retrievers[name].load()
        if self.topk is None:    
            self._topk = topk    
        else:                    
            assert self._topk == topk 
    
class OllamaDspy(Guesser):
    def __init__(self, params: Parameters):
        self.filename = params.ollama_guesser_filename
        self._qa_metric = validate_answer
        self._confidence_metric = lambda x, y: (x - y) ** 2
        self._model = CalibratedRAG()
        self._optimized = None

        if params.ollama_guesser_opt_algorithm == 'MIPROv2':
            from dspy.teleprompt import MIPROv2
            self._teleprompter = MIPROv2(
                metric=validate_answer,
                num_threads=params.ollama_guesser_threads,
                auto='medium',
            )
        elif params.ollama_guesser_opt_algorithm == 'BootstrapFewShotWithRandomSearch':
            from dspy.teleprompt import BootstrapFewShotWithRandomSearch
            self._teleprompter = BootstrapFewShotWithRandomSearch(
                metric=validate_answer,
                #max_labeled_demos=64,
                #max_rounds=1,
                #max_bootstrapped_demos=16,
                num_threads=params.ollama_guesser_threads)
            
    @staticmethod
    def create_dataset(questions, answers):
        return [dspy.Example(question=x, answer=y).with_inputs("question") for x, y in zip(questions, answers)]

    def setup_retrieval(self, name:str, filename:str, topk:int):
        """
        Add a new retriever to the system.

        Parameters:
        - name: The key of the retriever
        - filename: Where to load the retriever from (filename)
        - topk: How many retrieved results to get from the retriever per query
        """
        self._model.init_retriever(name, filename, topk)
    
    def train(self, training_data, answer_field='page', split_by_sentence=True,
                    min_length=-1, max_length=-1, remove_missing_pages=True):
        Guesser.train(self, training_data, answer_field, split_by_sentence, min_length,
                      max_length, remove_missing_pages)

        dataset = self.create_dataset(self.questions, self.answers)
        
        self._optimized = self._teleprompter.compile(
            self._model,
            trainset=dataset,
        )

    def __call__(self, question, max_n_guesses=1):
        if self._optimized == None:
            model = self._model
        else:
            model = self._optimized

        assert max_n_guesses == 1, "Can only give one guess"
        result = model(question)
        return [{"guess": result.best_guess, "confidence": result.confidence, "evidence": result.reasoning}]
            
            
    def set_ollama_model(self, model):
        self._lm = dspy.LM('ollama_chat/%s' % model, api_base='http://localhost:11434', api_key='')
        dspy.configure(lm=self._lm)

    def load(self):
        path = self.filename
        self._model.load("%s.json" % path)        

    def save(self):
        path = self.filename
        if self._optimized is None:
            self._model.save("%s.json" % path)
        else:
            self._optimized.save("%s.json" % path)
        
if __name__ == "__main__":
    import argparse
    import gzip
    import json
    
    from guesser import kTOY_DATA

    from dspy.evaluate import Evaluate

    from nltk.tokenize import sent_tokenize

    from parameters import load_questions, add_guesser_params, add_general_params, add_question_params, setup_logging, instantiate_guesser

    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='Question Type')
    add_general_params(parser)
    guesser_params = add_guesser_params(parser)
    add_question_params(parser)

    flags = parser.parse_args()
    guesser_params["ollama"].load_command_line_params(flags)

    questions = load_questions(flags)
    od = instantiate_guesser("Ollama", flags, False)

    # First, let's show some examples
    for qq in kTOY_DATA["train"]:
        result =  od(qq["text"])
        print(qq, result)

    # TODO: can we use some of the more standard question flags for this?
    with gzip.open(flags.ollama_guesser_dev_data) as infile:
        questions = json.load(infile)
        
        dev = od.create_dataset([sent_tokenize(x["text"])[:-2] for x in questions[:flags.ollama_guesser_validation_ex]],
                                [x["page"] for x in questions[:flags.ollama_guesser_validation_ex]])

        evaluator = Evaluate(devset=dev, num_threads=flags.ollama_guesser_threads,
                             display_progress=True, display_table=10, provide_traceback=True)

        print("PRETRAIN: %f" % evaluator(od._model, metric=validate_answer))

    
    od.train(questions[flags.ollama_guesser_validation_ex:])
    od.save()

    print("POSTTRAIN: %f" % evaluator(od._optimized, metric=validate_answer))
