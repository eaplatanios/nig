import logging
import nig
import os
import random

logger = logging.getLogger(__name__)

__author__ = 'Otilia Stretcu'


def generate_semantical_sentence(type_to_arguments, predicates):
    """
    Generates sentences represented as triples (predicate, agent, patient).
    Args:
        type_to_arguments:
        predicates:

    Returns:

    """
    verb = random.choice(list(predicates.keys()))
    roles = predicates[verb]
    agent_type = random.choice(roles['agent']) \
        if isinstance(roles['agent'], list) else roles['agent']
    agent = random.choice(type_to_arguments[agent_type])
    patient_type = random.choice(roles['patient']) \
        if isinstance(roles['patient'], list) else roles['patient']
    patient = random.choice(type_to_arguments[patient_type])

    return verb, agent, patient


def get_semantics(semantics_file, working_dir='.'):
    """
    Reads the semantical information from a yaml file.
    Args:
        semantics_file:
        working_dir:

    Returns:

    """
    configurations = nig.load_yaml(
        path=os.path.join(working_dir, semantics_file), ordered=False)
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

