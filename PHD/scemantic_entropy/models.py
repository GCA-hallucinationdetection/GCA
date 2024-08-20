"""Different prompts (and their logic) used to check for semantic equivalence."""
import logging
from collections import defaultdict
import numpy as np

import utils
from utils import (predict_w_log,  md5hash, cluster_assignment_entropy)


class SpoofData:
    def __getitem__(self, item):
        return f'<{item}>'


class BaseModel:

    def __init__(
            self, *, n_questions, n_regenerate, n_stochastic_questions):
        super().__init__()
        self.n_questions = n_questions
        self.n_regenerate = n_regenerate
        self.n_stochastic_questions = n_stochastic_questions

        # Dict of dict of list.
        self.export_predictions = defaultdict(defaultdict(list).copy)

    def predict_w_log(self, prompt, qidx):
        prediction = predict_w_log(prompt)
        self.export_predictions[qidx][md5hash(prompt)].append(prediction)
        return prediction

    def gen_facts(self, data):
        del data
        return 'Please list the specific factual propositions included in the answer above. Be complete and do not leave any factual claims out. Provide each claim as a separate sentence in a separate bullet point.'

    def get_all_prompts(self):
        # Spoof data to log prompting format.
        data = SpoofData()
        prompts = dict(
            gen_facts=self.gen_facts(data),
            base_gen_questions=self.base_gen_questions(data),
            base_answer_question=self.base_answer_question(data),
            base_equivalence=self.base_equivalence(data))

        return prompts

    def base_gen_questions(self, data):
        del data
        raise

    def base_answer_question(self, data):
        del data
        raise

    def base_equivalence(self, data):
        del data
        raise


class QAEquivalent(BaseModel):
    """Questions from context with ground truth answer with. Short answers with context. LLM Entailment without context."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def base_gen_questions(self, data):
        instruction = f"Generate a list of {self.n_questions} questions, that might have generated the sentence in the context of the preceding original text, as well as their answers. Please do not use specific facts that appear in the follow-up sentence when formulating the question.\nMake the questions and answers diverse. Avoid yes-no questions.\nThe answers should not be a full sentence and as short as possible, e.g. only a name, place, or thing. Use the format \"1. {{question}} -- {{answer}}\""

        if data['text_so_far'] is None:
            return f"""You see the sentence:

{data["proposition"]}

{instruction}"""
        else:
            return f"""Following this text:

{data["text_so_far"]}

You see the sentence:

{data["proposition"]}

{instruction}"""

    def base_answer_question(self, data):

        instruction = 'Please answer this question. Do not answer in a full sentence. Answer with as few words as possible, e.g. only a name, place, or thing.'

        if data['text_so_far'] is None:
            return f"""We are writing an answer to the question "{data["user_question"]}". First, we observe the following question:

{data["question"]}

{instruction}"""
        else:
            return f"""We are writing an answer to the question "{data["user_question"]}". So far we have written:

{data["text_so_far"]}

The next sentence should be the answer to the following question:

{data["question"]}

{instruction}"""

    def base_equivalence(self, data):
        prompt = 'Are the following answers equivalent?'
        for i in range(1, self.n_regenerate + 2):
            prompt += f'\nPossible Answer {i}: ' + '{}'
        prompt += '\nRespond only with "yes" or "no".'

        return prompt.format(
            data['expected_answers'], *data['regen_answers'])

    def check_truth(self, *, rp, data):
        gen_questions_prompt = self.base_gen_questions(data)
        expected_answers, questions = [], []
        for _ in range(self.n_stochastic_questions):
            success = False
            while not success:
                try:
                    gen_questions = self.predict_w_log(gen_questions_prompt, 2).split('\n')
                    expected_answers.extend([q.split(' -- ')[1] for q in gen_questions if q])
                    questions.extend([q[3:].split(' -- ')[0] for q in gen_questions if q])
                    success = True
                except Exception as e:
                    logging.warning('Retrying `gen_questions`, failed with error: %s', e)

        print(f'Extracted questions: {questions}')
        print(f'Extracted expected answers: {expected_answers}')

        uncertainties = []
        for qidx, (expected_answer, question) in enumerate(zip(expected_answers, questions)):
            print(f'Regenerate answers for question {qidx} "{question}":')

            regen_answers = []
            # << ANSWER EACH QUESTION MULTIPLE TIMES >>
            fdata = {**data, 'question': question}
            for _ in range(self.n_regenerate):
                answer_prompt = self.base_answer_question(fdata)
                answer = self.predict_w_log(answer_prompt, 3)
                regen_answers.append(answer)

            # << CHECK IF ANSWERS ARE EQUIVALENT >>
            if self.__class__.__name__ in ['QADebertaEntailment', 'QALLMEntailment']:

                answers = [expected_answer, *regen_answers]
                clusters, uncertainty = self.get_semantic_uncertainty(answers, fdata)

                # Account for GPT refusal to answer questions.
                stop_words = ['not available', 'not provided', 'unknown', 'unclear']
                unknown_count = 0
                for answer in answers:
                    if answer != None:
                        for stop_word in stop_words:
                            if stop_word in answer.lower():
                                unknown_count += 1
                                break
                if unknown_count >= len(answers) // 2:
                    logging.warning('Not answerable, setting uncertainty to maximum.')
                    uncertainty = -np.log(1 / len(answers))
                    clusters = str(clusters) + ' not answerable!'

                print(f'Semantic Clustering Input: {answers}')
                print(f'Semantic Clustering Output: {clusters}, uncertainty: {uncertainty}')
                equiv_response = clusters

            else:
                equiv_prompt = self.base_equivalence({
                    'expected_answers': expected_answer,
                    'regen_answers': regen_answers})
                equiv_response = self.predict_w_log(equiv_prompt, 3)
                uncertainty = utils.get_yes_no(equiv_response)

            uncertainties.append(uncertainty)

            rp[f'question-{qidx}'] = dict(
                question=question,
                answers=regen_answers,
                expected_answer=expected_answer,
                equiv_response=equiv_response,
                uncertainty=uncertainty,
            )


        return np.mean(uncertainties)



class QALLMEntailment(QAEquivalent):

    def get_all_prompts(self):
        # Spoof data to log prompting format.
        data = SpoofData()
        prompts = dict(
            gen_facts=self.gen_facts(data),
            base_gen_questions=self.base_gen_questions(data),
            base_answer_question=self.base_answer_question(data),
            base_equivalence=self.base_equivalence(data))

        return prompts

    def base_equivalence(self, data):

        prompt = f"""We are writing an answer to the question "{data["user_question"]}"."""

        if data['text_so_far'] is None:
            prompt = prompt + f""" First, we are trying to answer the subquestion "{data["question"]}".\n"""
        else:
            prompt = prompt + f""" So far we have written:

{data["text_so_far"]}

Next, we are trying to answer the subquestion "{data["question"]}".
Does at least one of the following two possible answers entail the other?

Possible Answer 1: {data["text1"]}
Possible Answer 2: {data["text2"]}

Respond with yes or no."""

        return prompt

    def are_equivalent(self, text1, text2, data):

        if text1 == text2:
            print(f'Skip entailment check: {text1} == {text2}.')
            return True

        equivalence_prompt = self.base_equivalence({'text1': text1, 'text2': text2, **data})
        equivalence = self.predict_w_log(equivalence_prompt, 3)
        uncertainty = utils.get_yes_no(equivalence)

        # If yes in equivalence --> uncertainty == 0 --> return True.
        return {0: True, 1: False}[uncertainty]

    def get_semantic_uncertainty(self, answers, fdata):
        semantic_ids = self.get_semantic_ids(answers, fdata)
        uncertainty = cluster_assignment_entropy(semantic_ids)
        return semantic_ids, uncertainty

    def get_semantic_ids(self, strings_list, data):
        """Group list of predictions into semantic meaning."""
        # Initialise all ids with -1.
        semantic_set_ids = [-1] * len(strings_list)
        # Keep track of current id.
        next_id = 0
        for i, string1 in enumerate(strings_list):
            # Check if string1 already has an id assigned.
            if semantic_set_ids[i] == -1:
                # If string1 has not been assigned an id, assign it next_id.
                semantic_set_ids[i] = next_id
                for j in range(i + 1, len(strings_list)):
                    # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                    if self.are_equivalent(string1, strings_list[j], data):
                        semantic_set_ids[j] = next_id
                next_id += 1
        assert -1 not in semantic_set_ids
        return semantic_set_ids


all_models = dict(
    QAEquivalent=QAEquivalent,
    QALLMEntailment=QALLMEntailment,
)
