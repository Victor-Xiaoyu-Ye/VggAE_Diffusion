import copy
import unittest
from scripts.verify_domain_cache import validate_stats


class DomainCacheTests(unittest.TestCase):
    def test_reject_incomplete_changed_and_missing_rgb(self):
        s={'num_samples':128,'num_failed':0,'num_shards':48,'config':{
            'csv_sha256':'fixed','clips_per_video':1,'decode_retries':0,
            'independent_anchor':True,'window_ae_norm':'legacy','store_rgb':True}}
        validate_stats(s,'fixed',128,1)
        for key,value in [('csv_sha256','other'),('decode_retries',8),('store_rgb',False),('clips_per_video',4)]:
            bad=copy.deepcopy(s);bad['config'][key]=value
            with self.assertRaises(ValueError):validate_stats(bad,'fixed',128,1)
        for key,value in [('num_samples',127),('num_failed',1)]:
            bad=copy.deepcopy(s);bad[key]=value
            with self.assertRaises(ValueError):validate_stats(bad,'fixed',128,1)
        s['num_samples']=8192;s['config']['clips_per_video']=4;s['num_shards']=47
        with self.assertRaises(ValueError):validate_stats(s,'fixed',8192,4,True)


if __name__=='__main__':unittest.main()
