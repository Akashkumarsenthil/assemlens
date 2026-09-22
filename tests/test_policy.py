import unittest
from assemlens.policy import Evidence, Route, StepVerifier, route

class PolicyTests(unittest.TestCase):
    def test_blur_never_escalates(self):
        self.assertEqual(route(Evidence(True,False,3,True,True,10)).route, Route.RECAPTURE)
    def test_cloud_needs_all_gates(self):
        self.assertEqual(route(Evidence(True,True,2,True,True,1)).route, Route.CLOUD)
        for e in [Evidence(True,True,2,False,True,1), Evidence(True,True,2,True,False,1), Evidence(True,True,2,True,True,0)]:
            self.assertEqual(route(e).route, Route.ABSTAIN)
    def test_unsupported_product(self):
        self.assertEqual(route(Evidence(False,True,2,True,True,1)).route, Route.ABSTAIN)
    def test_clear_supported_local(self):
        self.assertEqual(route(Evidence(True,True)).route, Route.LOCAL)
    def test_duplicate_capture_cannot_verify(self):
        v=StepVerifier('s1'); self.assertFalse(v.observe('s1','a','complete'))
        self.assertFalse(v.observe('s1','a','complete'))
        with self.assertRaises(ValueError): v.advance('s2')
        self.assertTrue(v.observe('s1','b','complete')); v.advance('s2')
        self.assertFalse(v.ready)
    def test_uncertainty_resets_evidence(self):
        v=StepVerifier('s1'); v.observe('s1','a','complete'); v.observe('s1','b','uncertain')
        self.assertFalse(v.observe('s1','c','complete'))
        self.assertTrue(v.observe('s1','d','complete'))
    def test_wrong_step_and_invalid_verdict(self):
        v=StepVerifier('s1')
        with self.assertRaises(ValueError): v.observe('s2','a','complete')
        with self.assertRaises(ValueError): v.observe('s1','a','probably')

if __name__=='__main__': unittest.main()
