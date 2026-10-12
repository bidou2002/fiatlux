from copy import deepcopy
import json
import unittest

from fiatlux_app.graph import (add_visualization, check_connections, compile_graph, graph_examples,
                               scientific_fingerprint, upgrade, validate_graph)
from fiatlux_app.model import ConfigurationError, examples


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.graph = graph_examples()['04_canvas']

    def test_roundtrip_and_order_ignore_positions_and_node_array_order(self):
        expected = compile_graph(self.graph)
        self.graph['nodes'].reverse()
        for index, pose in enumerate(self.graph['layout']['positions'].values()):
            pose['x'], pose['y'] = -index*300, index*70
        self.assertEqual(compile_graph(json.loads(json.dumps(self.graph))), expected)
        self.assertEqual(scientific_fingerprint(self.graph), scientific_fingerprint(graph_examples()['04_canvas']))

    def test_v1_upgrade_keeps_scientific_chain(self):
        chain = examples()['01_circular']
        self.assertEqual(compile_graph(upgrade(chain)), chain)

    def test_multiple_observers_do_not_affect_scientific_fingerprint(self):
        expected = scientific_fingerprint(self.graph)
        add_visualization(self.graph, 'second-phase', 'atmosphere', 'field', 'phase', 300, 650, '0')
        self.assertEqual(scientific_fingerprint(self.graph), expected)
        self.graph['nodes'][-1]['params'].update(colormap='gray', minimum=-1., maximum=1.)
        self.assertEqual(scientific_fingerprint(self.graph), expected)

    def test_typed_ports_and_scientific_branch_are_rejected(self):
        graph = deepcopy(self.graph)
        graph['connections'][0]['from']['port'] = 'opd'
        with self.assertRaisesRegex(ConfigurationError, 'ports incompatibles'):
            check_connections(graph)
        graph = deepcopy(self.graph)
        graph['connections'].append({'from': {'node':'pupil','port':'field'},'to': {'node':'detector','port':'in'}})
        with self.assertRaisesRegex(ConfigurationError, 'entrées scientifiques|branche scientifique'):
            check_connections(graph)

    def test_cycle_is_rejected(self):
        graph = deepcopy(self.graph)
        graph['connections'] = [e for e in graph['connections'] if e['to']['node'] not in ('pupil','focus')]
        graph['connections'].append({'from':{'node':'dm','port':'field'},'to':{'node':'pupil','port':'in'}})
        with self.assertRaisesRegex(ConfigurationError, 'Cycle scientifique'):
            check_connections(graph)

    def test_grid_mismatch_reports_connection(self):
        self.graph['nodes'][1]['params']['grid'] = 'focal'
        with self.assertRaisesRegex(ConfigurationError, 'source → pupil.*grille'):
            check_connections(self.graph)

    def test_unconnected_draft_is_not_executed(self):
        self.graph['connections'].pop(1)
        with self.assertRaisesRegex(ConfigurationError, 'chaîne scientifique incomplète'):
            compile_graph(self.graph)

    def test_visualization_requires_actual_opd_port_and_valid_display(self):
        self.graph['nodes'][-1]['params'].update(quantity='opd')
        with self.assertRaisesRegex(ConfigurationError, 'quantité incompatible'):
            check_connections(self.graph)
        self.graph = graph_examples()['04_canvas']
        self.graph['nodes'][-1]['params'].update(minimum=2, maximum=1)
        with self.assertRaisesRegex(ConfigurationError, 'limite basse'):
            validate_graph(self.graph)


if __name__ == '__main__':
    unittest.main()
