from copy import deepcopy
import threading
import time
import unittest
from fiatlux_app.graph import graph_examples
from fiatlux_app.revisions import RevisionManager


class Result:
    def __init__(self, config):
        self.value=config['spectrum']['magnitude']
    def metadata(self):
        return {'magnitude':self.value}


class Engine:
    def __init__(self, started, release):
        self.started, self.release, self.calls=started, release, []
    def affected(self, config):
        return ['dm','focus','detector']
    def execute(self, config):
        self.calls.append(config['spectrum']['magnitude'])
        if len(self.calls)==1:
            self.started.set();self.release.wait(5)
        return Result(config)


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.started,self.release=threading.Event(),threading.Event()
        self.engine=Engine(self.started,self.release)
        self.manager=RevisionManager(lambda:self.engine)
        self.graph=graph_examples()['04_canvas']
    def tearDown(self):
        self.release.set();self.manager.close()
    def wait_current(self, revision):
        until=time.monotonic()+5
        while time.monotonic()<until:
            state=self.manager.status('test')
            if state['published_revision']==revision:return state
            time.sleep(.005)
        self.fail('Revision not published')
    def test_coalesces_latest_and_never_publishes_old_work(self):
        self.manager.submit('test',1,self.graph)
        self.assertTrue(self.started.wait(2))
        for revision in (2,3):
            self.graph['spectrum']['magnitude']=revision
            self.manager.submit('test',revision,self.graph)
        self.release.set()
        state=self.wait_current(3)
        self.assertEqual(state['result']['magnitude'],3)
        self.assertEqual(self.engine.calls,[0.,3])
        self.assertEqual(self.manager.submit('test',2,self.graph)['accepted'],False)
        self.assertEqual(self.manager.status('test')['published_revision'],3)
    def test_invalid_configuration_supersedes_running_work_without_execution(self):
        self.manager.submit('test',1,self.graph)
        self.assertTrue(self.started.wait(2))
        self.graph['nodes'][3]['params']['commands']=[float('nan')]
        state=self.manager.submit('test',2,self.graph)
        self.assertFalse(state['valid'])
        self.release.set()
        self.manager.executor.shutdown(wait=True)
        state=self.manager.status('test')
        self.assertEqual(state['state'],'error')
        self.assertEqual(state['published_revision'],-1)
        self.assertEqual(len(self.engine.calls),1)
    def test_invalid_latest_keeps_last_image_snapshot(self):
        self.release.set()
        self.manager.submit('test',1,self.graph)
        self.wait_current(1)
        self.graph['connections']=[]
        self.manager.submit('test',2,self.graph)
        state=self.manager.status('test')
        self.assertEqual(state['published_revision'],1)
        self.assertEqual(state['result']['magnitude'],0.)
        self.assertEqual(state['state'],'error')


if __name__ == '__main__':unittest.main()
