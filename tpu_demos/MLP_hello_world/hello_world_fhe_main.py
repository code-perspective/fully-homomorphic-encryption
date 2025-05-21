r"""An experimental file to get an xprof link for running a simple FHE ML program.

It can be launched via XManager. Use the following command:

google_xmanager launch "$(pwd)/privacy/private_computing/fhe/jaxite/xmanager/launcher.py" -- \
--platform=ghostlite_pod \
--topology=2x2 \
--xm_resource_alloc="group:core-dynamic/pss-shared-data-resources-gqm-acl" \
--build_target_path='//experimental/phazon/fhe_on_cloud/hello_world:hello_world_fhe_xprof.par'
"""

import timeit
from typing import Sequence

from jaxite.jaxite_bool import bool_params
from jaxite.jaxite_bool import jaxite_bool
from jaxite.jaxite_bool import type_converters
from jaxite.jaxite_lib import random_source
from jaxite.jaxite_lib import types

import hello_world_fhe_lib


MILLISECONDS_PER_SECOND = 1000


def encrypt_int(
    x: int,
    num_bits: int,
    lwe_rng: random_source.PseudorandomSource,
    cks: jaxite_bool.ClientKeySet,
) -> types.LweCiphertext:
  """encrypt an int."""
  cleartext_x = type_converters.uint_to_bit_slice(x, num_bits)
  ciphertext_x = [jaxite_bool.encrypt(z, cks, lwe_rng) for z in cleartext_x]
  return ciphertext_x


def decrypt_int(
    ciphertext: list[types.LweCiphertext],
    num_bits: int,
    cks: jaxite_bool.ClientKeySet,
) -> int:
  """Decrypt the output."""
  return type_converters.bit_slice_to_uint(
      [jaxite_bool.decrypt(z, cks) for z in ciphertext], num_bits
  )


def run_hello_world() -> None:
  """Runs the and_gate benchmark."""

  int_size = 8
  msg_a = 5

  boolean_params = bool_params.get_params_for_128_bit_security()
  lwe_rng = bool_params.get_lwe_rng_for_128_bit_security(1)
  rlwe_rng = bool_params.get_rlwe_rng_for_128_bit_security(1)

  cks = jaxite_bool.ClientKeySet(boolean_params, lwe_rng, rlwe_rng)
  sks = jaxite_bool.ServerKeySet(cks, boolean_params, lwe_rng, rlwe_rng)

  ciphertext_a = encrypt_int(msg_a, 8, lwe_rng, cks)

  # Calling function once before profiling it so compile-time doesn't get
  # included in timing metircs and xprof.
  ciphertext_c = hello_world_fhe_lib.main(
      ciphertext_a, sks, boolean_params
  )

  def timed_fn():
    hello_world_fhe_lib.main(ciphertext_a, sks, boolean_params)

  timer = timeit.Timer(timed_fn)
  millis = timer.repeat(1, 1) * MILLISECONDS_PER_SECOND
  
  print('runtime: %s ms', sum(millis))

  result = decrypt_int(ciphertext_c, int_size, cks)

  print('hello world result: %d', result)



if __name__ == '__main__':
  run_hello_world()