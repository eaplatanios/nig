import nig
import os
import random
import sys


def extract_sentence_data(working_dir, config_file):
    configurations = nig.load_yaml(
        path=os.path.join(working_dir, config_file), ordered=False)
    # For each 'type' of argument, make a map from type to the arguments.
    type_to_arguments = {type_name: [] for type_name in configurations['type']}
    for argument_details in configurations['argument']:
        for argument, types in argument_details.items():
            for type_name in types:
                type_to_arguments[type_name].append(argument)
    #argument_to_types = {k: v for argument in configurations['argument']
    #    for k, v in argument.items()}
    predicates = {k: v for predicate in configurations['predicate']
        for k, v in predicate.items()}
    return type_to_arguments, predicates


def generate_sentences(type_to_arguments, predicates, num_sentences_per_verb=1):
    active_sentences = []
    passive_sentences = []
    for verb, roles in predicates.items():
        for sentence_index in range(num_sentences_per_verb):
            agent_type = random.choice(roles['agent']) \
                if isinstance(roles['agent'], list) else roles['agent']
            agent = random.choice(type_to_arguments[agent_type])
            patient_type = random.choice(roles['patient']) \
                if isinstance(roles['patient'], list) else roles['patient']
            patient = random.choice(type_to_arguments[patient_type])
            active_sentences.append(
                agent + ' ' + roles['active'] + ' ' + patient + '.')
            passive_sentences.append(
                patient + ' was ' + roles['passive'] + ' by ' + agent + '.')

    return active_sentences, passive_sentences


if len(sys.argv) > 1:
    working_dir = os.path.join(os.getcwd(), sys.argv[1])
else:
    working_dir = os.getcwd()
config_file = sys.argv[2] if len(sys.argv) > 2 else 'sentences_template.yaml'

type_to_arguments, predicates = extract_sentence_data(working_dir, config_file)
active_sentences, passive_sentences = generate_sentences(type_to_arguments, predicates)

print('------- Active sentences ---------')
for sentence in active_sentences:
    print(sentence)

print('------- Passive sentences ---------')
for sentence in passive_sentences:
    print(sentence)
