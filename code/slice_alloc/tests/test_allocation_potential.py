"""Native reward shaping must telescope, with no change to success objective."""
import unittest
from slice_alloc import guided_allocation as guided
from slice_alloc import allocation_final as base


class PotentialTests(unittest.TestCase):
    def test_native_potential_preserves_complete_return(self):
        env=guided.GuidedEnv(rule_parameters=(2.,.5,0.),shaping=.5)
        try:
            obs,_=env.reset(options={'scenario':base.scenario(229003,24)})
            done=False;reward=0.;native=0.;different=False
            self.assertEqual(env.potential(),0.)
            while not done:
                action=guided.rule_action('modal_greedy',env,(2.,.5,0.))
                obs,r,done,_,info=env.step(action)
                reward+=r;native+=info['native_reward'];different|=r!=info['native_reward']
            self.assertTrue(different)
            self.assertEqual(env.potential(),0.)
            self.assertEqual(reward,native)
            self.assertEqual(reward,env.result()['success'])
        finally:env.close()


if __name__=='__main__':unittest.main()
