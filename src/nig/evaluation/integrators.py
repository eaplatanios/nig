# Copyright 2016, The NIG Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from __future__ import absolute_import, division, print_function

import abc
import errno
import logging
import os
import subprocess

from six import with_metaclass

from nig.evaluation import integrator_pb2
from nig.evaluation.constraints import Constraint

__author__ = 'eaplatanios'

__all__ = ['silently_remove', 'Integrator', 'MajorityVoteIntegrator',
           'AgreementIntegrator', 'BayesianIntegrator',
           'CoupledBayesianIntegrator', 'HierarchicalCoupledBayesianIntegrator',
           'LogicIntegrator', 'save_error_rates', 'load_error_rates',
           'save_error_rates_to_protobin', 'load_error_rates_from_protobin',
           'save_error_rates_to_csv', 'load_error_rates_from_csv',
           'save_predicted_instances', 'load_predicted_instances',
           'save_predicted_instances_to_protobin',
           'load_predicted_instances_from_protobin',
           'save_predicted_instances_to_csv',
           'load_predicted_instances_from_csv', 'save_observed_instances',
           'load_observed_instances', 'save_observed_instances_to_protobin',
           'load_observed_instances_from_protobin',
           'save_observed_instances_to_csv', 'load_observed_instances_from_csv',
           'get_filename_extension', 'save_constraints', 'load_constraints']

_builder_format = 'makina.learn.classification.reflection.{}$Builder'.format

logger = logging.getLogger(__name__)


def silently_remove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise


class Integrator(with_metaclass(abc.ABCMeta, object)):
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def options(self):
        pass

    @abc.abstractmethod
    def java_obj(self, predicted, observed=None, constraints=None, seed=None):
        pass

    def run(self, predicted, observed=None, constraints=None,
            integrate_data=False, seed=None, use_cli=False, working_dir='.',
            makina_jar=None, use_csv=False, clean_up=True, jvm_options=None):
        if not isinstance(constraints, list):
            constraints = [constraints]
        if makina_jar is None:
            evaluation_dir = os.path.dirname(__file__)
            makina_jar = os.path.join(evaluation_dir, 'makina.jar')
        if not os.path.isfile(makina_jar):
            raise OSError('makina.jar not found!')
        if use_cli:
            return self._run_command_line(predicted, observed,
                                          constraints, integrate_data, seed,
                                          working_dir, makina_jar, use_csv,
                                          clean_up, jvm_options)
        else:
            return self._run_jnius(predicted, observed, constraints,
                                   integrate_data, seed, makina_jar,
                                   jvm_options)

    def _run_jnius(self, predicted, observed, constraints, integrate_data,
                   seed, makina_jar, jvm_options):
        if jvm_options is None:
            jvm_options = ['-Xmx4G']
        import jnius_config
        try:
            jnius_config.add_options(*jvm_options)
            jnius_config.set_classpath(makina_jar)
        except ValueError:
            pass
        integrator = self.java_obj(_predicted_to_java(predicted),
                                   _observed_to_java(observed),
                                   constraints=constraints, seed=seed)
        error_rates = _error_rates_from_java(integrator.errorRates())
        return error_rates if not integrate_data \
            else (error_rates,
                  _predicted_from_java(integrator.integratedData()))

    def _run_command_line(self, predicted, observed, constraints,
                          integrate_data, seed, working_dir, makina_jar,
                          use_csv, clean_up, jvm_options):
        files = {}
        file_ext = '.csv' if use_csv else '.protobin'
        files['constraints'] = os.path.join(working_dir, 'constraints.txt')
        for name in {'predicted', 'observed', 'error_rates', 'integrated'}:
            files[name] = os.path.join(working_dir, name + file_ext)
        save_predicted_instances(predicted, files['predicted'])
        if observed is not None:
            save_observed_instances(observed, files['observed'])
        if jvm_options is None:
            jvm_options = ['-Xmx4G']
        cli_options = ['java', '-cp', makina_jar]
        cli_options.extend(jvm_options)
        cli_options.extend(
            ['makina.learn.classification.reflection.Integrator',
             '-d', files['predicted'],
             '-e', files['error_rates'],
             '-m', self.name()])
        if observed is not None:
            cli_options.extend(['-od', files['observed']])
        if self.options():
            cli_options.extend(['-o', ':'.join(self.options())])
        if constraints is not None:
            save_constraints(constraints, files['constraints'])
            cli_options.extend(['-c', files['constraints']])
        if integrate_data:
            cli_options.extend(['-i', files['integrated']])
        if seed is not None:
            cli_options.extend(['-s', seed])
        return_code = subprocess.Popen(cli_options).wait()
        if return_code != 0:
            logger.error('An error occurred while running the command '
                         '{}.'.format(' '.join(cli_options)))
            return None
        result = load_error_rates([], files['error_rates'])
        if integrate_data:
            integrated = load_predicted_instances([], files['integrated'])
            result = (result, integrated)
        if clean_up:
            for name in files.keys():
                silently_remove(files[name])
        return result


class MajorityVoteIntegrator(Integrator):
    def name(self):
        return 'MVI'

    def options(self):
        return []

    def java_obj(self, predicted, observed=None, constraints=None, seed=None):
        from jnius import autoclass
        builder = autoclass(_builder_format('MajorityVoteIntegrator'))
        return builder(predicted).build()


class AgreementIntegrator(Integrator):
    def __init__(self, highest_order=-1, only_even_orders=True):
        self.highest_order = highest_order
        self.only_even_orders = only_even_orders

    def name(self):
        return 'AI'

    def options(self):
        return [str(self.highest_order), '1' if self.only_even_orders else '0']

    def java_obj(self, predicted, observed=None, constraints=None, seed=None):
        from jnius import autoclass
        builder = autoclass(_builder_format('AgreementIntegrator'))
        return builder(predicted) \
            .highestOrder(self.highest_order) \
            .onlyEvenCardinalitySubsetsAgreements(self.only_even_orders) \
            .build()


class BayesianIntegrator(Integrator):
    def __init__(self, number_of_burn_in_samples=4000,
                 number_of_thinning_samples=10, number_of_samples=200,
                 labels_prior_alpha=1.0, labels_prior_beta=1.0,
                 error_rates_prior_alpha=1.0,
                 error_rates_prior_beta=2.0):
        self.number_of_burn_in_samples = number_of_burn_in_samples
        self.number_of_thinning_samples = number_of_thinning_samples
        self.number_of_samples = number_of_samples
        self.labels_prior_alpha = labels_prior_alpha
        self.labels_prior_beta = labels_prior_beta
        self.error_rates_prior_alpha = error_rates_prior_alpha
        self.error_rates_prior_beta = error_rates_prior_beta

    def name(self):
        return 'BI'

    def options(self):
        return [str(self.number_of_burn_in_samples),
                str(self.number_of_thinning_samples),
                str(self.number_of_samples),
                str(self.labels_prior_alpha),
                str(self.labels_prior_beta),
                str(self.error_rates_prior_alpha),
                str(self.error_rates_prior_beta)]

    def java_obj(self, predicted, observed=None, constraints=None, seed=None):
        from jnius import autoclass
        builder = autoclass(_builder_format('BayesianIntegrator'))
        return builder(predicted) \
            .numberOfBurnInSamples(self.number_of_burn_in_samples) \
            .numberOfThinningSamples(self.number_of_thinning_samples) \
            .numberOfSamples(self.number_of_samples) \
            .labelsPriorAlpha(self.labels_prior_alpha) \
            .labelsPriorBeta(self.labels_prior_beta) \
            .errorRatesPriorAlpha(self.error_rates_prior_alpha) \
            .errorRatesPriorBeta(self.error_rates_prior_beta) \
            .seed(seed) \
            .build()


class CoupledBayesianIntegrator(Integrator):
    def __init__(self, number_of_burn_in_samples=4000,
                 number_of_thinning_samples=10, number_of_samples=200,
                 alpha=1.0, labels_prior_alpha=1.0, labels_prior_beta=1.0,
                 error_rates_prior_alpha=1.0, error_rates_prior_beta=2.0):
        self.number_of_burn_in_samples = number_of_burn_in_samples
        self.number_of_thinning_samples = number_of_thinning_samples
        self.number_of_samples = number_of_samples
        self.alpha = alpha
        self.labels_prior_alpha = labels_prior_alpha
        self.labels_prior_beta = labels_prior_beta
        self.error_rates_prior_alpha = error_rates_prior_alpha
        self.error_rates_prior_beta = error_rates_prior_beta

    def name(self):
        return 'CBI'

    def options(self):
        return [str(self.number_of_burn_in_samples),
                str(self.number_of_thinning_samples),
                str(self.number_of_samples),
                str(self.alpha),
                str(self.labels_prior_alpha),
                str(self.labels_prior_beta),
                str(self.error_rates_prior_alpha),
                str(self.error_rates_prior_beta)]

    def java_obj(self, predicted, observed=None, constraints=None, seed=None):
        from jnius import autoclass
        builder = autoclass(_builder_format('CoupledBayesianIntegrator'))
        return builder(predicted) \
            .numberOfBurnInSamples(self.number_of_burn_in_samples) \
            .numberOfThinningSamples(self.number_of_thinning_samples) \
            .numberOfSamples(self.number_of_samples) \
            .alpha(self.alpha) \
            .labelsPriorAlpha(self.labels_prior_alpha) \
            .labelsPriorBeta(self.labels_prior_beta) \
            .errorRatesPriorAlpha(self.error_rates_prior_alpha) \
            .errorRatesPriorBeta(self.error_rates_prior_beta) \
            .seed(seed) \
            .build()


class HierarchicalCoupledBayesianIntegrator(Integrator):
    def __init__(self, number_of_burn_in_samples=4000,
                 number_of_thinning_samples=10, number_of_samples=200,
                 alpha=1.0, gamma=1.0, labels_prior_alpha=1.0,
                 labels_prior_beta=1.0, error_rates_prior_alpha=1.0,
                 error_rates_prior_beta=2.0):
        self.number_of_burn_in_samples = number_of_burn_in_samples
        self.number_of_thinning_samples = number_of_thinning_samples
        self.number_of_samples = number_of_samples
        self.alpha = alpha
        self.gamma = gamma
        self.labels_prior_alpha = labels_prior_alpha
        self.labels_prior_beta = labels_prior_beta
        self.error_rates_prior_alpha = error_rates_prior_alpha
        self.error_rates_prior_beta = error_rates_prior_beta

    def name(self):
        return 'HCBI'

    def options(self):
        return [str(self.number_of_burn_in_samples),
                str(self.number_of_thinning_samples),
                str(self.number_of_samples),
                str(self.alpha),
                str(self.gamma),
                str(self.labels_prior_alpha),
                str(self.labels_prior_beta),
                str(self.error_rates_prior_alpha),
                str(self.error_rates_prior_beta)]

    def java_obj(self, predicted, observed=None, constraints=None, seed=None):
        from jnius import autoclass
        builder = autoclass(
            _builder_format('HierarchicalCoupledBayesianIntegrator'))
        return builder(predicted) \
            .numberOfBurnInSamples(self.number_of_burn_in_samples) \
            .numberOfThinningSamples(self.number_of_thinning_samples) \
            .numberOfSamples(self.number_of_samples) \
            .alpha(self.alpha) \
            .gamma(self.gamma) \
            .labelsPriorAlpha(self.labels_prior_alpha) \
            .labelsPriorBeta(self.labels_prior_beta) \
            .errorRatesPriorAlpha(self.error_rates_prior_alpha) \
            .errorRatesPriorBeta(self.error_rates_prior_beta) \
            .seed(seed) \
            .build()


class LogicIntegrator(Integrator):
    def name(self):
        return 'LI'

    def options(self):
        return []

    def java_obj(self, predicted, observed=None, constraints=None, seed=None):
        from jnius import autoclass
        builder = autoclass(_builder_format('LogicIntegrator'))
        hash_set_class = autoclass('java.util.HashSet')
        java_constraints = hash_set_class()
        if constraints is not None:
            for constraint in constraints:
                java_constraints.add(constraint.java_obj())
        return builder(predicted, observed) \
            .addConstraints(java_constraints) \
            .build()


def _predicted_to_java(predicted_instances):
    if predicted_instances is None:
        return None
    from jnius import autoclass
    array_list_class = autoclass('java.util.ArrayList')
    instances = array_list_class()
    label_class = autoclass('makina.learn.classification.Label')
    predicted_instance_class = autoclass('makina.learn.classification'
                                         '.reflection'
                                         '.Integrator$Data$PredictedInstance')
    for instance in predicted_instances:
        instances.add(predicted_instance_class(int(instance[0]),
                                               label_class(str(instance[1])),
                                               int(instance[2]),
                                               float(instance[3])))
    data_class = autoclass('makina.learn.classification'
                           '.reflection.Integrator$Data')
    return data_class(instances)


def _observed_to_java(observed_instances):
    if observed_instances is None:
        return None
    from jnius import autoclass
    array_list_class = autoclass('java.util.ArrayList')
    instances = array_list_class()
    label_class = autoclass('makina.learn.classification.Label')
    observed_instance_class = autoclass('makina.learn.classification'
                                        '.reflection'
                                        '.Integrator$Data$ObservedInstance')
    for instance in observed_instances:
        instances.add(observed_instance_class(int(instance[0]),
                                              label_class(str(instance[1])),
                                              int(instance[2])))
    data_class = autoclass('makina.learn.classification'
                           '.reflection.Integrator$Data')
    return data_class(instances)


def _iterate_java(java_iterator):
    while java_iterator.hasNext():
        yield java_iterator.next()


def _error_rates_from_java(java_error_rates):
    error_rates = []
    for java_error_rate in _iterate_java(java_error_rates.iterator()):
        error_rates.append((java_error_rate.label().name(),
                            java_error_rate.functionId(),
                            java_error_rate.value()))
    return error_rates


def _predicted_from_java(java_instances):
    predicted_instances = []
    for java_predicted_instance in _iterate_java(java_instances.iterator()):
        predicted_instances.append((java_predicted_instance.id(),
                                    java_predicted_instance.label().name(),
                                    java_predicted_instance.functionId(),
                                    java_predicted_instance.value()))
    return predicted_instances


def _observed_from_java(java_instances):
    observed_instances = []
    for java_observed_instance in _iterate_java(java_instances.iterator()):
        observed_instances.append((java_observed_instance.id(),
                                   java_observed_instance.label().name(),
                                   java_observed_instance.value()))
    return observed_instances


def save_error_rates(error_rates, filename):
    file_extension = get_filename_extension(filename)
    if file_extension == 'protobin':
        save_error_rates_to_protobin(error_rates, filename)
    elif file_extension == 'csv':
        save_error_rates_to_csv(error_rates, filename)
    else:
        raise ValueError('Unsupported file extension: ' + file_extension + '.')


def load_error_rates(error_rates, filename):
    file_extension = get_filename_extension(filename)
    if file_extension == 'protobin':
        return load_error_rates_from_protobin(error_rates, filename)
    elif file_extension == 'csv':
        return load_error_rates_from_csv(error_rates, filename)
    else:
        raise ValueError('Unsupported file extension: ' + file_extension + '.')


def save_error_rates_to_protobin(error_rates, filename):
    proto_error_rates = integrator_pb2.ErrorRates()
    for error_rate in error_rates:
        proto_error_rate = proto_error_rates.errorRate.add()
        proto_error_rate.label = str(error_rate[0])
        proto_error_rate.functionId = int(error_rate[1])
        proto_error_rate.value = float(error_rate[2])
    f = open(filename, 'wb')
    f.write(proto_error_rates.SerializeToString())
    f.close()


def load_error_rates_from_protobin(error_rates, filename):
    proto_error_rates = integrator_pb2.ErrorRates()
    f = open(filename, 'rb')
    proto_error_rates.ParseFromString(f.read())
    f.close()
    for proto_error_rate in proto_error_rates.errorRate:
        error_rates.append((proto_error_rate.label,
                            proto_error_rate.functionId,
                            proto_error_rate.value))
    return error_rates


def save_error_rates_to_csv(error_rates, filename):
    f = open(filename, 'w')
    f.write('LABEL,FUNCTION_ID,VALUE\n')
    for error_rate in error_rates:
        f.write(error_rate[0] + ',' + str(error_rate[1]) + ',' +
                str(error_rate[2]) + '\n')
    f.close()


def load_error_rates_from_csv(error_rates, filename):
    f = open(filename, 'r')
    lines = [line.rstrip('\n') for line in f]
    f.close()
    for line in lines:
        if line == 'LABEL,FUNCTION_ID,VALUE':
            continue
        parts = line.split(',')
        error_rates.append((parts[0], int(parts[1]), float(parts[2])))
    return error_rates


def save_predicted_instances(predicted_instances, filename):
    file_extension = get_filename_extension(filename)
    if file_extension == 'protobin':
        save_predicted_instances_to_protobin(predicted_instances, filename)
    elif file_extension == 'csv':
        save_predicted_instances_to_csv(predicted_instances, filename)
    else:
        raise ValueError('Unsupported file extension: ' + file_extension + '.')


def load_predicted_instances(predicted_instances, filename):
    file_extension = get_filename_extension(filename)
    if file_extension == 'protobin':
        return load_predicted_instances_from_protobin(predicted_instances,
                                                      filename)
    elif file_extension == 'csv':
        return load_predicted_instances_from_csv(predicted_instances, filename)
    else:
        raise ValueError('Unsupported file extension: ' + file_extension + '.')


def save_predicted_instances_to_protobin(predicted_instances, filename):
    proto_instances = integrator_pb2.PredictedInstances()
    for predicted_instance in predicted_instances:
        proto_predicted_instance = proto_instances.predictedInstance.add()
        proto_predicted_instance.id = int(predicted_instance[0])
        proto_predicted_instance.label = str(predicted_instance[1])
        proto_predicted_instance.functionId = int(predicted_instance[2])
        proto_predicted_instance.value = float(predicted_instance[3])
    f = open(filename, 'wb')
    f.write(proto_instances.SerializeToString())
    f.close()


def load_predicted_instances_from_protobin(predicted_instances, filename):
    proto_instances = integrator_pb2.PredictedInstances()
    f = open(filename, 'rb')
    proto_instances.ParseFromString(f.read())
    f.close()
    for proto_predicted_instance in proto_instances.predictedInstance:
        predicted_instances.append((proto_predicted_instance.id,
                                    proto_predicted_instance.label,
                                    proto_predicted_instance.functionId,
                                    proto_predicted_instance.value))
    return predicted_instances


def save_predicted_instances_to_csv(predicted_instances, filename):
    f = open(filename, 'w')
    f.write('ID,LABEL,FUNCTION_ID,VALUE\n')
    for instance in predicted_instances:
        f.write(str(instance[0]) + ',' + instance[1] + ',' +
                str(instance[2]) + ',' + str(instance[3]) + '\n')
    f.close()


def load_predicted_instances_from_csv(predicted_instances, filename):
    f = open(filename, 'r')
    lines = [line.rstrip('\n') for line in f]
    f.close()
    for line in lines:
        if line == 'ID,LABEL,FUNCTION_ID,VALUE':
            continue
        parts = line.split(',')
        predicted_instances.append((int(parts[0]), parts[1], int(parts[2]),
                                    float(parts[3])))
    return predicted_instances


def save_observed_instances(observed_instances, filename):
    file_extension = get_filename_extension(filename)
    if file_extension == 'protobin':
        save_observed_instances_to_protobin(observed_instances, filename)
    elif file_extension == 'csv':
        save_observed_instances_to_csv(observed_instances, filename)
    else:
        raise ValueError('Unsupported file extension: ' + file_extension + '.')


def load_observed_instances(observed_instances, filename):
    file_extension = get_filename_extension(filename)
    if file_extension == 'protobin':
        return load_observed_instances_from_protobin(observed_instances,
                                                     filename)
    elif file_extension == 'csv':
        return load_observed_instances_from_csv(observed_instances, filename)
    else:
        raise ValueError('Unsupported file extension: ' + file_extension + '.')


def save_observed_instances_to_protobin(observed_instances, filename):
    proto_instances = integrator_pb2.ObservedInstances()
    for observed_instance in observed_instances:
        proto_observed_instance = proto_instances.observedInstance.add()
        proto_observed_instance.id = int(observed_instance[0])
        proto_observed_instance.label = str(observed_instance[1])
        proto_observed_instance.value = int(observed_instance[2])  # TODO: Shouldn't this be float?
    f = open(filename, 'wb')
    f.write(proto_instances.SerializeToString())
    f.close()


def load_observed_instances_from_protobin(observed_instances, filename):
    proto_instances = integrator_pb2.ObservedInstances()
    f = open(filename, 'rb')
    proto_instances.ParseFromString(f.read())
    f.close()
    for proto_observed_instance in proto_instances.observedInstance:
        observed_instances.append((proto_observed_instance.id,
                                   proto_observed_instance.label,
                                   proto_observed_instance.functionId,
                                   proto_observed_instance.value))
    return observed_instances


def save_observed_instances_to_csv(observed_instances, filename):
    f = open(filename, 'w')
    f.write('ID,LABEL,VALUE\n')
    for instance in observed_instances:
        f.write(str(instance[0]) + ',' + instance[1] + ',' +
                ('1' if instance[2] else '0') + '\n')
    f.close()


def load_observed_instances_from_csv(observed_instances, filename):
    f = open(filename, 'r')
    lines = [line.rstrip('\n') for line in f]
    f.close()
    for line in lines:
        if line == 'ID,LABEL,VALUE':
            continue
        parts = line.split(',')
        observed_instances.append((int(parts[0]), parts[1], parts[2] == '1'))
    return observed_instances


def get_filename_extension(filename):
    return os.path.splitext(filename)[1][1:]


def save_constraints(constraints, filename):
    f = open(filename, 'w')
    for constraint in constraints:
        f.write(str(constraint) + '\n')
    f.close()


def load_constraints(constraints, filename):
    f = open(filename, 'r')
    lines = [line.rstrip('\n') for line in f]
    f.close()
    for line in lines:
        parsed_constraint = Constraint.from_str(line)
        if type(parsed_constraint) is list:
            constraints.extend(parsed_constraint)
        else:
            constraints.add(parsed_constraint)
    return constraints
