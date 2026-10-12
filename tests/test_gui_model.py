"""Data-only tests also runnable without PyTorch: python -m unittest ..."""
from copy import deepcopy
import unittest
from fiatlux_app.model import ConfigurationError, dumps, examples, loads, validate


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config = deepcopy(examples()['01_circular'])

    def test_saved_examples_roundtrip_without_mutation(self):
        for config in examples().values():
            original = deepcopy(config)
            self.assertEqual(loads(dumps(config)), config)
            self.assertEqual(config, original)

    def test_errors_refer_to_offending_block(self):
        self.config['blocks'][1]['params']['grid'] = 'focal'
        with self.assertRaisesRegex(ConfigurationError, 'pupil.*grille incompatible'):
            validate(self.config)

    def test_unknown_classes_and_expressions_are_data_not_code(self):
        self.config['blocks'][1]['type'] = '__import__'
        with self.assertRaisesRegex(ConfigurationError, 'non pris en charge'):
            validate(self.config)
        self.config = deepcopy(examples()['01_circular'])
        self.config['blocks'][1]['params']['radius'] = '__import__("os").system("echo unsafe")'
        with self.assertRaisesRegex(ConfigurationError, 'nombre fini'):
            validate(self.config)

    def test_nonfinite_values_and_duplicate_json_keys(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            self.config['spectrum']['magnitude'] = value
            with self.assertRaises(ConfigurationError):
                validate(self.config)
        with self.assertRaisesRegex(ConfigurationError, 'non finie'):
            loads('{"format": NaN}')
        with self.assertRaisesRegex(ConfigurationError, 'dupliquée'):
            loads('{"format":"a","format":"b"}')

    def test_required_fields_version_and_order(self):
        for mutate in (lambda c: c.pop('precision'), lambda c: c.update(version=2),
                       lambda c: c['blocks'].reverse(),
                       lambda c: c['blocks'][1]['params'].update(extra=1)):
            candidate = deepcopy(self.config)
            mutate(candidate)
            with self.assertRaises(ConfigurationError):
                validate(candidate)

    def test_fresnel_parameters_and_zero_distance(self):
        prop = self.config['blocks'][2]['params']
        prop['propagation'] = 'fresnel'
        with self.assertRaisesRegex(ConfigurationError, 'Fresnel exige'):
            validate(self.config)
        prop.update(focal_length=None, distance=0)
        with self.assertRaisesRegex(ConfigurationError, 'distance zéro'):
            validate(self.config)
        prop['output_grid'] = 'pupil'
        self.config['blocks'][-1]['params']['grid'] = 'pupil'
        validate(self.config)

    def test_fft_monochromatic_and_natural_grid(self):
        prop = self.config['blocks'][2]
        prop['type'] = 'FFTPropagator'
        prop['params'].pop('output_grid')
        self.config['spectrum']['samples'] = 2
        with self.assertRaisesRegex(ConfigurationError, 'monochromatique'):
            validate(self.config)
        self.config['spectrum']['samples'] = 1
        with self.assertRaisesRegex(ConfigurationError, 'grille incompatible'):
            validate(self.config)
        grid = self.config['grids']['pupil']
        spacing = self.config['spectrum']['central_wavelength']*10/(grid['nx']*grid['dx'])
        self.config['grids']['focal'] = dict(nx=64, ny=64, dx=spacing, dy=spacing)
        validate(self.config)

    def test_dm_has_no_silent_saturation_or_square_conversion(self):
        config = deepcopy(examples()['02_elt_dm'])
        config['blocks'][2]['params']['commands'][0] = 2e-6
        with self.assertRaisesRegex(ConfigurationError, 'stroke'):
            validate(config)
        config['blocks'][2]['params']['commands'][0] = 0
        config['grids']['pupil']['ny'] = 64
        with self.assertRaisesRegex(ConfigurationError, 'carrée'):
            validate(config)


if __name__ == '__main__':
    unittest.main()
