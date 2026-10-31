"""TP profiling follows executed models, not auxiliary constructor imports."""

import gc
import unittest
import weakref

from h3_ascend import profiling


class Model:
    pass


class ModelRegistry(unittest.TestCase):
    def setUp(self):
        self.previous = profiling._FORWARD_MODELS[:]
        profiling._FORWARD_MODELS.clear()

    def tearDown(self):
        profiling._FORWARD_MODELS[:] = self.previous

    def test_executed_deserialized_model_takes_precedence(self):
        constructed, executed = Model(), Model()
        profiling.register_live_model(executed)
        self.assertEqual(profiling._select_live_models([weakref.ref(constructed)]), [executed])

    def test_duplicate_registration_is_one_model(self):
        model = Model()
        profiling.register_live_model(model)
        profiling.register_live_model(model)
        self.assertEqual(profiling._select_live_models([]), [model])

    def test_dead_model_does_not_keep_worker_storage_alive(self):
        model = Model()
        profiling.register_live_model(model)
        reference = weakref.ref(model)
        del model
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(profiling._select_live_models([]), [])

    def test_empty_helper_and_constructor_fallback(self):
        self.assertEqual(profiling._select_live_models([]), [])
        model = Model()
        self.assertEqual(profiling._select_live_models([weakref.ref(model)]), [model])
