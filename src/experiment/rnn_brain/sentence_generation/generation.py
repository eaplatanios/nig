from .semantics import get_semantics, generate_semantical_sentence
from .syntax import Grammar, Lexicon, generate_voiced_syntactical_sentence

import logging
import os
import random
import sys

logger = logging.getLogger(__name__)


def match_semantics_syntax(semantic_concepts, syntactic_roles, lexicon):
    def _match(concept, syntactic_role):
        if syntactic_role[0] == 'V':
            if concept not in lexicon.verbs:
                return None
            if 'singular' in syntactic_role[1]:
                return lexicon.verbs[concept]['active'] \
                    if 'active' in lexicon.verbs[concept] else None
            if 'plural' in syntactic_role[1]:
                return lexicon.verbs[concept]['root'] \
                    if 'root' in lexicon.verbs[concept] else None
            if 'participle' in syntactic_role[1]:
                return lexicon.verbs[concept]['passive'] \
                    if 'passive' in lexicon.verbs[concept] else None
        elif syntactic_role[0] == 'NP':
            if concept not in lexicon.nouns:
                return None
            return concept if syntactic_role[1] < lexicon.nouns[concept] \
                else None

        return None

    match = [_match(concept, role)
             for concept, role in zip(semantic_concepts, syntactic_roles)]

    return None if None in match else match


def syntactic_roles_to_semantic_match(syntactical_sentence,
                                      voice_is_active=True):
    """
    Selects which elements of the syntactical sentence must match the
    verb, agent and patient, respectively.
    Args:
        syntactical_sentence: a list of tuples (symbol, attributes)
        voice_is_active(bool): If True, we assume the syntactical sentence is in
            active voice, therefore the agent must match the first noun phrase,
            and the patient the second noun phrase. If False, the patient must
            match the first noun phrase, and the agent the second noun phrase.
    Returns:

    """
    first_NP = None
    first_NP_index = None
    second_NP = None
    second_NP_index = None
    first_V = None
    first_V_index = None
    index = -1
    for symbol, attributes in syntactical_sentence:
        index = index + 1
        if symbol == 'NP':
            if first_NP is None:
                first_NP = (symbol, attributes)
                first_NP_index = index
            elif second_NP is None:
                second_NP = (symbol, attributes)
                second_NP_index = index
        elif symbol == "V" and first_V is None:
            first_V = (symbol, attributes)
            first_V_index = index

    if voice_is_active:
        return (first_V, first_NP, second_NP), \
               (first_V_index, first_NP_index, second_NP_index)
    return (first_V, second_NP, first_NP), \
           (first_V_index, second_NP_index, first_NP_index)


def generate_sentence(type_to_arguments, predicates, grammar, lexicon,
                      voice_is_active=True):
    def pattern_match_sentence(
            semantic_concepts, syntactical_sentence, indexes_in_sentence):
        sentence = [symbol for symbol, _ in syntactical_sentence]
        for index, concept in zip(indexes_in_sentence, semantic_concepts):
            sentence[index] = concept
        return sentence

    # Generate semantic sentence.
    semantic_sentence = generate_semantical_sentence(
        type_to_arguments, predicates)

    logger.debug('\n\n----------------')
    logger.debug('Voice: %s' % 'active' if voice_is_active else 'passive')
    logger.debug(
        'Semantic sentence(verb, agent, patient): %s' % str(semantic_sentence))

    # Generate a syntactical sentence that will match the semantical sentence.
    match = None
    count = 0
    while match is None:
        syntactical_sentence = generate_voiced_syntactical_sentence(
            grammar, is_active=voice_is_active, count=count)
        syntactic_roles, indexes_in_sentence = \
            syntactic_roles_to_semantic_match(syntactical_sentence,
                                              voice_is_active=voice_is_active)
        match = match_semantics_syntax(
            semantic_sentence, syntactic_roles, lexicon)
        # if count > 5000000:
        #    print('\n\n\n')
        count += 1

    logger.debug('Syntactical sentence: %s' % str(syntactical_sentence))
    logger.debug('Match: %s' % str(match))

    # Put together the final sentence as a list of words.
    sentence = pattern_match_sentence(
        match, syntactical_sentence, indexes_in_sentence)
    logger.debug('Sentence after pattern match: %s' % str(sentence))

    return sentence


def sentence_to_text(words):
    return words[0].capitalize() + ' ' + ' '.join(words[1:]) + '.'


if len(sys.argv) > 1:
    working_dir = os.path.join(os.getcwd(), sys.argv[1])
else:
    working_dir = os.path.join(os.getcwd(), 'sentence_generation')
syntax_file = sys.argv[2] if len(sys.argv) > 2 else 'syntax.yaml'
semantics_file = sys.argv[3] if len(sys.argv) > 3 else 'semantics.yaml'
lexicon_file = sys.argv[4] if len(sys.argv) > 4 else 'lexicon.yaml'

# Generate the semantics.
type_to_arguments, predicates = get_semantics(semantics_file, working_dir)

# Read the syntax file and create a grammar.
grammar = Grammar(syntax_file, working_dir)

# Read the lexicon file and create a Lexicon.
lexicon = Lexicon(lexicon_file, working_dir)

# Connect semantics and syntax.
sentences = []
for i in range(1000):
    sentences.append(generate_sentence(
        type_to_arguments, predicates, grammar, lexicon,
        voice_is_active=random.choice([True, False])))
    print(sentence_to_text(sentences[-1]))

with open(os.path.join(working_dir, 'generated_sentences.txt'), 'w') as f:
    f.write('\n'.join([sentence_to_text(s) for s in sentences]))
