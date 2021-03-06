# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: makina/learn/classification/reflection/integrator.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='makina/learn/classification/reflection/integrator.proto',
  package='makina.learn.classification.reflection',
  serialized_pb=_b('\n7makina/learn/classification/reflection/integrator.proto\x12&makina.learn.classification.reflection\"<\n\x10ObservedInstance\x12\n\n\x02id\x18\x01 \x02(\x05\x12\r\n\x05label\x18\x02 \x02(\t\x12\r\n\x05value\x18\x03 \x02(\x08\"g\n\x11ObservedInstances\x12R\n\x10observedInstance\x18\x01 \x03(\x0b\x32\x38.makina.learn.classification.reflection.ObservedInstance\"Q\n\x11PredictedInstance\x12\n\n\x02id\x18\x01 \x02(\x05\x12\r\n\x05label\x18\x02 \x02(\t\x12\x12\n\nfunctionId\x18\x03 \x02(\x05\x12\r\n\x05value\x18\x04 \x02(\x01\"j\n\x12PredictedInstances\x12T\n\x11predictedInstance\x18\x01 \x03(\x0b\x32\x39.makina.learn.classification.reflection.PredictedInstance\"=\n\tErrorRate\x12\r\n\x05label\x18\x01 \x02(\t\x12\x12\n\nfunctionId\x18\x02 \x02(\x05\x12\r\n\x05value\x18\x03 \x02(\x01\"R\n\nErrorRates\x12\x44\n\terrorRate\x18\x01 \x03(\x0b\x32\x31.makina.learn.classification.reflection.ErrorRateB\x12\x42\x10IntegratorProtos')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_OBSERVEDINSTANCE = _descriptor.Descriptor(
  name='ObservedInstance',
  full_name='makina.learn.classification.reflection.ObservedInstance',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='makina.learn.classification.reflection.ObservedInstance.id', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='label', full_name='makina.learn.classification.reflection.ObservedInstance.label', index=1,
      number=2, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='value', full_name='makina.learn.classification.reflection.ObservedInstance.value', index=2,
      number=3, type=8, cpp_type=7, label=2,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=99,
  serialized_end=159,
)


_OBSERVEDINSTANCES = _descriptor.Descriptor(
  name='ObservedInstances',
  full_name='makina.learn.classification.reflection.ObservedInstances',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='observedInstance', full_name='makina.learn.classification.reflection.ObservedInstances.observedInstance', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=161,
  serialized_end=264,
)


_PREDICTEDINSTANCE = _descriptor.Descriptor(
  name='PredictedInstance',
  full_name='makina.learn.classification.reflection.PredictedInstance',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='makina.learn.classification.reflection.PredictedInstance.id', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='label', full_name='makina.learn.classification.reflection.PredictedInstance.label', index=1,
      number=2, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='functionId', full_name='makina.learn.classification.reflection.PredictedInstance.functionId', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='value', full_name='makina.learn.classification.reflection.PredictedInstance.value', index=3,
      number=4, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=266,
  serialized_end=347,
)


_PREDICTEDINSTANCES = _descriptor.Descriptor(
  name='PredictedInstances',
  full_name='makina.learn.classification.reflection.PredictedInstances',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='predictedInstance', full_name='makina.learn.classification.reflection.PredictedInstances.predictedInstance', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=349,
  serialized_end=455,
)


_ERRORRATE = _descriptor.Descriptor(
  name='ErrorRate',
  full_name='makina.learn.classification.reflection.ErrorRate',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='label', full_name='makina.learn.classification.reflection.ErrorRate.label', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='functionId', full_name='makina.learn.classification.reflection.ErrorRate.functionId', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='value', full_name='makina.learn.classification.reflection.ErrorRate.value', index=2,
      number=3, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=457,
  serialized_end=518,
)


_ERRORRATES = _descriptor.Descriptor(
  name='ErrorRates',
  full_name='makina.learn.classification.reflection.ErrorRates',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='errorRate', full_name='makina.learn.classification.reflection.ErrorRates.errorRate', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=520,
  serialized_end=602,
)

_OBSERVEDINSTANCES.fields_by_name['observedInstance'].message_type = _OBSERVEDINSTANCE
_PREDICTEDINSTANCES.fields_by_name['predictedInstance'].message_type = _PREDICTEDINSTANCE
_ERRORRATES.fields_by_name['errorRate'].message_type = _ERRORRATE
DESCRIPTOR.message_types_by_name['ObservedInstance'] = _OBSERVEDINSTANCE
DESCRIPTOR.message_types_by_name['ObservedInstances'] = _OBSERVEDINSTANCES
DESCRIPTOR.message_types_by_name['PredictedInstance'] = _PREDICTEDINSTANCE
DESCRIPTOR.message_types_by_name['PredictedInstances'] = _PREDICTEDINSTANCES
DESCRIPTOR.message_types_by_name['ErrorRate'] = _ERRORRATE
DESCRIPTOR.message_types_by_name['ErrorRates'] = _ERRORRATES

ObservedInstance = _reflection.GeneratedProtocolMessageType('ObservedInstance', (_message.Message,), dict(
  DESCRIPTOR = _OBSERVEDINSTANCE,
  __module__ = 'makina.learn.classification.reflection.integrator_pb2'
  # @@protoc_insertion_point(class_scope:makina.learn.classification.reflection.ObservedInstance)
  ))
_sym_db.RegisterMessage(ObservedInstance)

ObservedInstances = _reflection.GeneratedProtocolMessageType('ObservedInstances', (_message.Message,), dict(
  DESCRIPTOR = _OBSERVEDINSTANCES,
  __module__ = 'makina.learn.classification.reflection.integrator_pb2'
  # @@protoc_insertion_point(class_scope:makina.learn.classification.reflection.ObservedInstances)
  ))
_sym_db.RegisterMessage(ObservedInstances)

PredictedInstance = _reflection.GeneratedProtocolMessageType('PredictedInstance', (_message.Message,), dict(
  DESCRIPTOR = _PREDICTEDINSTANCE,
  __module__ = 'makina.learn.classification.reflection.integrator_pb2'
  # @@protoc_insertion_point(class_scope:makina.learn.classification.reflection.PredictedInstance)
  ))
_sym_db.RegisterMessage(PredictedInstance)

PredictedInstances = _reflection.GeneratedProtocolMessageType('PredictedInstances', (_message.Message,), dict(
  DESCRIPTOR = _PREDICTEDINSTANCES,
  __module__ = 'makina.learn.classification.reflection.integrator_pb2'
  # @@protoc_insertion_point(class_scope:makina.learn.classification.reflection.PredictedInstances)
  ))
_sym_db.RegisterMessage(PredictedInstances)

ErrorRate = _reflection.GeneratedProtocolMessageType('ErrorRate', (_message.Message,), dict(
  DESCRIPTOR = _ERRORRATE,
  __module__ = 'makina.learn.classification.reflection.integrator_pb2'
  # @@protoc_insertion_point(class_scope:makina.learn.classification.reflection.ErrorRate)
  ))
_sym_db.RegisterMessage(ErrorRate)

ErrorRates = _reflection.GeneratedProtocolMessageType('ErrorRates', (_message.Message,), dict(
  DESCRIPTOR = _ERRORRATES,
  __module__ = 'makina.learn.classification.reflection.integrator_pb2'
  # @@protoc_insertion_point(class_scope:makina.learn.classification.reflection.ErrorRates)
  ))
_sym_db.RegisterMessage(ErrorRates)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('B\020IntegratorProtos'))
# @@protoc_insertion_point(module_scope)
