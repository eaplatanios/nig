import nig
import os
import random
import re

__author__ = 'Otilia Stretcu'

class Grammar:
    def __init__(self, syntax_file, working_dir='.', start_symbol=None,
                 start_attributes=frozenset({})):
        self.rules = self._read_syntax_from_file(
            filename=syntax_file, working_dir=working_dir)
        self.start_symbol = start_symbol if start_symbol is not None \
            else 'S'   # TODO: maybe detect start symbol from file
        assert self.start_symbol in self.rules, "The start symbol needs to be " \
                                                "provided in the rules as a non-terminal."

        self.start_attributes = start_attributes
        if self.start_attributes != frozenset({}):
            assert self.start_attributes in self.rules[start_symbol], \
                "The attrbutes provided for the start symbol must be provided" \
                " in the rules as a non-terminal."

        self.non_terminals, self.terminals = self._classify_symbols(
            start_symbol=self.start_symbol,
            start_attributes=self.start_attributes)

    def _read_syntax_from_file(self, filename, working_dir='.'):
        """
        Reads syntax information from a yaml file.
        Args:
            filename:
            working_dir:

        Returns:

        """
        def parse_pos_attributes(pos_attributes):
            """
            Separates the part of speech from the attributes.
            Args:
                pos_attributes(str): a string containing the POS and its attributes such
                    as: 'NP[singular,vowel]'

            Returns:
                POS(str): the part of speech
                attributes(frozenset of strings): the attributes
            """
            elements = re.split(r'\[|\]|,', pos_attributes)
            elements = [elem for elem in elements if len(elem) > 0]
            return elements[0], frozenset(elements[1:])

        data = nig.load_yaml(
            path=os.path.join(working_dir, filename), ordered=False)
        rules = dict()
        for rule in data['rules']:
            for key, value in rule.items():
                symbol, attributes = parse_pos_attributes(key)
                production = [parse_pos_attributes(elem) for elem in value]
                if symbol not in rules:
                    rules[symbol] = {}
                if attributes in rules[symbol]:
                    rules[symbol][attributes].append(production)
                else:
                    rules[symbol][attributes] = [production]
        return rules

    def _classify_symbols(self, start_symbol=None,
                          start_attributes=frozenset({})):
        """

        Args:
            start_symbol:
            start_attributes: if None, it will consider all available attributes
                for the given start symbol.

        Returns:

        """
        start_symbol = start_symbol \
            if start_symbol is not None else self.start_symbol

        terminals = set([])
        non_terminals = set([])
        if start_symbol in self.rules and start_attributes is frozenset({}):
            # If the start attributes are not provided, we consider all
            # possible attributes for the given start symbol.
            for attributes in self.rules[start_symbol]:
                non_terminals_children, terminals_children = \
                    self._classify_symbols(start_symbol, attributes)
                terminals = terminals.union(terminals_children)
                non_terminals = non_terminals.union(non_terminals_children)
        elif start_symbol in self.rules and \
                        start_attributes in self.rules[start_symbol]:
            # If we find the combination of start_symbol and start_attributes
            # on the left hand side of the rules, we add them to the
            # non-terminals list of not already there.
            if (start_symbol, start_attributes) not in non_terminals:
                non_terminals.add((start_symbol, start_attributes))
                for rule in self.rules[start_symbol][start_attributes]:
                    for node in rule:
                        non_terminals_children, terminals_children = \
                            self._classify_symbols(start_symbol=node[0], start_attributes=node[1])
                        terminals = terminals.union(terminals_children)
                        non_terminals = non_terminals.union(non_terminals_children)
        elif (start_symbol not in self.rules) or \
                (start_attributes not in self.rules[start_symbol]):
            # If we do NOT find the combination of start_symbol and
            # start_attributes on the left hand side of the rules, we add them
            # to the terminals list of not already there.
            if (start_symbol, start_attributes) not in terminals:
                # TODO: check that they appear on the right hand side of some rule.
                terminals.add((start_symbol, start_attributes))

        return non_terminals, terminals


class Lexicon:
    def __init__(self, lexicon_file, working_dir='.'):
        self.nouns, self.verbs = \
            self._read_lexicon_from_file(lexicon_file, working_dir)

    def _read_lexicon_from_file(self, filename, working_dir='.'):
        data = nig.load_yaml(
            path=os.path.join(working_dir, filename), ordered=False)
        # Create the nouns list.
        nouns = dict()
        for noun in data['noun']:
            for key, value in noun.items():
                nouns[key] = frozenset(value)

        # Create the verbs list.
        verbs = dict()
        for verb_data in data['verb']:
            for verb, attributes in verb_data.items():
                verbs[verb] = attributes

        return nouns, verbs


def generate_voiced_syntactical_sentence(grammar, is_active=True, count=0):
    return generate_syntactical_sentence(
        grammar, start_symbol='S',
        start_attributes=frozenset({'active' if is_active else 'passive'}),
        count=count)


def generate_syntactical_sentence(
        grammar, start_symbol='S', start_attributes=frozenset({}),
        indentation="", count=0):
    # Check if the provided start symbol and attributes are valid terminals
    # in the grammar, and return them.
    if (start_symbol, start_attributes) in grammar.terminals or \
                    (start_symbol, frozenset({})) in grammar.terminals:
        return [(start_symbol, start_attributes)]

    # Check if the provided start symbol and attributes are valid non-terminals
    # in the grammar.
    if (start_symbol not in grammar.rules) or (start_attributes != frozenset({})
                                               and (start_symbol, start_attributes) not in grammar.non_terminals):
        raise ValueError('The provided start_symbol and start_attributes'
                         'are not valid non-terminals of the grammar.')

    symbol_attributes = [start_attributes]
    # If start_attributes is empty, then we consider all possible attributes of
    # the provided start symbol.
    if start_attributes is frozenset({}):
        symbol_attributes = list(grammar.rules[start_symbol].keys())

    # Sample one combination of (start_symbol, attributes).
    # start_attributes = random.choice(symbol_attributes)

    # Sample a production rule for the non-terminal (start_symbol, attributes).
    # production = random.choice(grammar.rules[start_symbol][start_attributes])

    production = random.choice([r for sa in symbol_attributes for r in grammar.rules[start_symbol][sa]])
    #if count > 5000000:
    #    print(indentation, (start_symbol, start_attributes), '-->', production, '-->')

    # Create a sentence by expanding all the elements in the production.
    sentence = [item for symbol_attribute in production
                for item in generate_syntactical_sentence(
            grammar, symbol_attribute[0], symbol_attribute[1],
            indentation=indentation + "   ")]
    #if count > 5000000:
    #    print(indentation, '-->', sentence)
    return sentence

